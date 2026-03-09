"""Tests for trading_agents.agents.risk_manager"""

import pytest
from unittest.mock import MagicMock
from trading_agents.agents.risk_manager import RiskManagerAgent
from trading_agents.core.llm import BaseLLM


@pytest.fixture
def risk_manager(mock_llm):
    mock_llm.generate_json.return_value = {
        "approved": True,
        "modified_size": 3,
        "risk_assessment": "moderate",
        "violations": [],
        "warnings": [],
        "stop_loss_price": None,
        "take_profit_price": None,
        "recommendation": "approve",
        "reasoning": "Risk within limits"
    }
    return RiskManagerAgent(
        llm=mock_llm,
        max_position_size=10.0,
        max_drawdown_pct=0.15
    )


class TestHardLimits:

    def test_rejects_oversized_position(self, risk_manager):
        decision = {'suggested_size': 15, 'decision': 'BUY'}
        portfolio = {'equity': 100000, 'cash': 100000, 'initial_capital': 100000, 'current_price': 100}
        position = {'has_position': False, 'size': 0}

        result = risk_manager._apply_hard_limits(decision, portfolio, position)
        assert result['approved'] is False
        assert any('exceeds max' in v for v in result['violations'])

    def test_rejects_on_drawdown_breach(self, risk_manager):
        decision = {'suggested_size': 3, 'decision': 'BUY'}
        portfolio = {'equity': 80000, 'cash': 80000, 'initial_capital': 100000, 'current_price': 100}
        position = {'has_position': False, 'size': 0}

        result = risk_manager._apply_hard_limits(decision, portfolio, position)
        assert result['approved'] is False
        assert result['modified_size'] == 0

    def test_approves_valid_trade(self, risk_manager):
        decision = {'suggested_size': 3, 'decision': 'BUY'}
        portfolio = {'equity': 100000, 'cash': 100000, 'initial_capital': 100000, 'current_price': 100}
        position = {'has_position': False, 'size': 0}

        result = risk_manager._apply_hard_limits(decision, portfolio, position)
        assert result['approved'] is True

    def test_drawdown_formula_consistent(self, risk_manager):
        """Test that drawdown is calculated as (initial - equity) / initial (positive = loss)."""
        decision = {'suggested_size': 3, 'decision': 'BUY'}
        # 10% loss: equity=90000, initial=100000 → drawdown=0.1
        portfolio = {'equity': 90000, 'cash': 90000, 'initial_capital': 100000, 'current_price': 100}
        position = {'has_position': False, 'size': 0}

        # 10% < 15% limit → should be approved
        result = risk_manager._apply_hard_limits(decision, portfolio, position)
        assert result['approved'] is True

    def test_insufficient_capital(self, risk_manager):
        decision = {'suggested_size': 5, 'decision': 'BUY'}
        # Use initial_capital close to equity so drawdown doesn't trigger first
        portfolio = {'equity': 95000, 'cash': 100, 'initial_capital': 100000, 'current_price': 100}
        position = {'has_position': False, 'size': 0}

        result = risk_manager._apply_hard_limits(decision, portfolio, position)
        assert result['approved'] is False
        assert any('Insufficient capital' in v for v in result['violations'])
