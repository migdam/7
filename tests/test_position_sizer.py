"""Tests for trading_agents.enhancements.position_sizer"""

import pytest
from trading_agents.enhancements.position_sizer import (
    PositionSizingEngine,
    FixedFractionalSizer,
    KellyCriterionSizer,
    ATRBasedSizer,
    VolatilityScaledSizer,
    OptimalFSizer,
    calculate_optimal_position
)


class TestFixedFractionalSizer:

    def test_basic_sizing(self):
        sizer = FixedFractionalSizer(fraction=0.02)
        size = sizer.calculate_size(capital=100000, price=100)
        assert size == pytest.approx(20.0)

    def test_different_fraction(self):
        sizer = FixedFractionalSizer(fraction=0.05)
        size = sizer.calculate_size(capital=100000, price=50)
        assert size == pytest.approx(100.0)


class TestKellyCriterionSizer:

    def test_positive_kelly(self):
        sizer = KellyCriterionSizer(fraction=1.0)
        size = sizer.calculate_size(
            capital=100000, price=100,
            win_rate=0.6, avg_win=100, avg_loss=50
        )
        assert size > 0

    def test_half_kelly(self):
        sizer_full = KellyCriterionSizer(fraction=1.0)
        sizer_half = KellyCriterionSizer(fraction=0.5)

        size_full = sizer_full.calculate_size(
            capital=100000, price=100,
            win_rate=0.6, avg_win=100, avg_loss=50
        )
        size_half = sizer_half.calculate_size(
            capital=100000, price=100,
            win_rate=0.6, avg_win=100, avg_loss=50
        )
        assert size_half == pytest.approx(size_full / 2)

    def test_zero_avg_loss(self):
        sizer = KellyCriterionSizer()
        size = sizer.calculate_size(capital=100000, price=100, avg_loss=0)
        assert size >= 0


class TestATRBasedSizer:

    def test_with_atr(self):
        sizer = ATRBasedSizer(risk_per_atr=2.0, max_risk_pct=0.02)
        size = sizer.calculate_size(capital=100000, price=100, atr=3.5)
        expected = (100000 * 0.02) / (3.5 * 2.0)
        assert size == pytest.approx(expected)

    def test_without_atr_uses_fallback(self):
        sizer = ATRBasedSizer()
        size = sizer.calculate_size(capital=100000, price=100)
        assert size > 0


class TestVolatilityScaledSizer:

    def test_inverse_volatility(self):
        sizer = VolatilityScaledSizer(target_volatility=0.15, base_allocation=0.1)
        size_low_vol = sizer.calculate_size(capital=100000, price=100, volatility=0.10)
        size_high_vol = sizer.calculate_size(capital=100000, price=100, volatility=0.30)
        assert size_low_vol > size_high_vol


class TestPositionSizingEngine:

    def test_zero_price_guard(self):
        engine = PositionSizingEngine(strategy='fixed_fractional', fraction=0.02)
        result = engine.calculate(capital=100000, price=0)
        assert result['size'] == 0.0

    def test_negative_price_guard(self):
        engine = PositionSizingEngine(strategy='fixed_fractional', fraction=0.02)
        result = engine.calculate(capital=100000, price=-10)
        assert result['size'] == 0.0

    def test_returns_metadata(self):
        engine = PositionSizingEngine(strategy='fixed_fractional', fraction=0.02)
        result = engine.calculate(capital=100000, price=100)
        assert 'size' in result
        assert 'position_value' in result
        assert 'position_pct' in result
        assert 'strategy' in result

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            PositionSizingEngine(strategy='nonexistent')


class TestConvenienceFunction:

    def test_calculate_optimal_position(self):
        size = calculate_optimal_position(capital=100000, price=100, strategy='fixed_fractional')
        assert size > 0
