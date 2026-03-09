"""Tests for trading_agents.enhancements.monte_carlo"""

import pytest
import pandas as pd
import numpy as np
from trading_agents.enhancements.monte_carlo import MonteCarloSimulator


def mock_backtest(data):
    """Simple mock backtest function."""
    returns = data['close'].pct_change().dropna()
    total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    return {
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
    }


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
    return pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


class TestBootstrapResample:

    def test_output_shape(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        resampled = mc.bootstrap_resample(sample_data)
        assert resampled.shape == sample_data.shape
        assert set(resampled.columns) == set(sample_data.columns)

    def test_ohlc_validity(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        resampled = mc.bootstrap_resample(sample_data)
        # High should be >= low
        assert (resampled['high'] >= resampled['low']).all()


class TestGBM:

    def test_output_shape(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        gbm_data = mc.geometric_brownian_motion(sample_data)
        assert gbm_data.shape == sample_data.shape

    def test_positive_prices(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        gbm_data = mc.geometric_brownian_motion(sample_data)
        assert (gbm_data['close'] > 0).all()


class TestRunSimulation:

    def test_sequential_bootstrap(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        results = mc.run_simulation(method='bootstrap')
        assert len(results) > 0
        assert len(mc.results) == 5

    def test_sequential_gbm(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5, random_seed=42)
        results = mc.run_simulation(method='gbm')
        assert len(results) > 0

    def test_unknown_method_raises(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=1)
        with pytest.raises(ValueError, match="Unknown method"):
            mc.run_simulation(method='invalid')

    def test_analyze_results(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=10, random_seed=42)
        mc.run_simulation(method='bootstrap')
        analysis = mc.analyze_results()
        assert 'total_return_pct' in analysis
        assert 'mean' in analysis['total_return_pct']
        assert 'std' in analysis['total_return_pct']

    def test_confidence_intervals(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=20, random_seed=42)
        mc.run_simulation(method='bootstrap')
        ci = mc.get_confidence_intervals('total_return_pct', confidence=0.95)
        assert 'mean' in ci
        assert 'lower_bound' in ci
        assert 'upper_bound' in ci
        assert ci['lower_bound'] <= ci['mean'] <= ci['upper_bound']

    def test_empty_results(self, sample_data):
        mc = MonteCarloSimulator(mock_backtest, sample_data, n_simulations=5)
        # Without running simulation
        assert mc.analyze_results() == {}
        assert mc.get_confidence_intervals('total_return_pct') == {}
