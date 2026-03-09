"""Tests for trading_agents.core.market_env"""

import pytest
import pandas as pd
import numpy as np
from trading_agents.core.market_env import MarketEnvironment, Position, Trade


class TestMarketEnvironmentValidation:

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="must not be empty"):
            MarketEnvironment(data=df)

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({'close': [100], 'volume': [1000]},
                          index=pd.date_range('2024-01-01', periods=1))
        with pytest.raises(ValueError, match="missing required columns"):
            MarketEnvironment(data=df)

    def test_accepts_valid_data(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        assert env.initial_capital == 100000.0


class TestMarketEnvironmentTrading:

    def test_step_advances_index(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        assert env.current_idx == 0
        assert env.step() is True
        assert env.current_idx == 1

    def test_step_returns_false_at_end(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        env.current_idx = len(sample_ohlcv_data) - 1
        assert env.step() is False

    def test_open_and_close_position(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data, initial_capital=100000)
        env.step()  # Move to idx 1

        # Open long position
        result = env.execute_trade(size=2.0, rationale="test buy")
        assert result['success'] is True
        assert result['action'] == 'OPEN'
        assert env.position is not None
        assert env.position.size == 2.0

        env.step()  # Move to idx 2

        # Close position
        result = env.execute_trade(size=0, rationale="test close")
        assert result['success'] is True
        assert result['action'] == 'CLOSE'
        assert env.position is None
        assert env.total_trades == 1

    def test_cannot_exceed_max_position(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data, max_position_size=5.0)
        env.step()
        result = env.execute_trade(size=10.0)
        assert result['success'] is False

    def test_get_equity(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data, initial_capital=100000)
        assert env.get_equity() == 100000.0

    def test_get_position_info_no_position(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        info = env.get_position_info()
        assert info['has_position'] is False
        assert info['size'] == 0.0

    def test_performance_metrics(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        for _ in range(10):
            env.step()
        metrics = env.get_performance_metrics()
        assert 'total_return_pct' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'sharpe_ratio' in metrics

    def test_reset(self, sample_ohlcv_data):
        env = MarketEnvironment(data=sample_ohlcv_data)
        env.step()
        env.step()
        env.reset()
        assert env.current_idx == 0
        assert env.cash == env.initial_capital
        assert env.position is None
        assert len(env.trades) == 0


class TestPosition:

    def test_unrealized_pnl(self):
        pos = Position(symbol="TEST", size=10, entry_price=100, entry_time=None, current_price=110)
        assert pos.unrealized_pnl == 100.0
        assert pos.unrealized_pnl_pct == pytest.approx(10.0)
        assert pos.market_value == 1100.0

    def test_zero_entry_price(self):
        pos = Position(symbol="TEST", size=1, entry_price=0, entry_time=None, current_price=100)
        assert pos.unrealized_pnl_pct == 0.0
