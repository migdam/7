"""Tests for trading_agents.data.indicators"""

import pytest
import pandas as pd
import numpy as np
from trading_agents.data.indicators import (
    add_all_indicators,
    get_indicator_summary,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_volatility,
    calculate_momentum,
    calculate_stochastic
)


class TestIndividualIndicators:

    def test_sma(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = calculate_sma(s, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)
        assert pd.isna(result.iloc[0])

    def test_ema(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = calculate_ema(s, 3)
        assert len(result) == 5
        assert not pd.isna(result.iloc[2])

    def test_rsi(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        result = calculate_rsi(s, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        result = calculate_macd(s)
        assert 'macd' in result
        assert 'macd_signal' in result
        assert 'macd_histogram' in result
        assert len(result['macd']) == 100

    def test_bollinger_bands(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(50)) + 100)
        result = calculate_bollinger_bands(s, 20)
        assert 'bb_upper' in result
        assert 'bb_middle' in result
        assert 'bb_lower' in result
        # Upper > middle > lower where not NaN
        valid_idx = result['bb_upper'].dropna().index
        assert (result['bb_upper'][valid_idx] >= result['bb_middle'][valid_idx]).all()
        assert (result['bb_middle'][valid_idx] >= result['bb_lower'][valid_idx]).all()

    def test_atr(self, sample_ohlcv_data):
        result = calculate_atr(sample_ohlcv_data, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_volatility(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        result = calculate_volatility(s, 20)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_momentum(self):
        s = pd.Series([100, 105, 110, 115, 120], dtype=float)
        result = calculate_momentum(s, 2)
        assert result.iloc[2] == pytest.approx(10.0)

    def test_stochastic(self, sample_ohlcv_data):
        result = calculate_stochastic(sample_ohlcv_data)
        assert 'stoch_k' in result
        assert 'stoch_d' in result
        valid_k = result['stoch_k'].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestAddAllIndicators:

    def test_adds_expected_columns(self, sample_ohlcv_data):
        result = add_all_indicators(sample_ohlcv_data)
        expected_cols = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volatility', 'momentum',
            'stoch_k', 'stoch_d'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self, sample_ohlcv_data):
        result = add_all_indicators(sample_ohlcv_data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns

    def test_does_not_modify_original(self, sample_ohlcv_data):
        original_cols = set(sample_ohlcv_data.columns)
        add_all_indicators(sample_ohlcv_data)
        assert set(sample_ohlcv_data.columns) == original_cols


class TestGetIndicatorSummary:

    def test_returns_expected_keys(self, sample_ohlcv_data):
        df = add_all_indicators(sample_ohlcv_data)
        summary = get_indicator_summary(df, 100)
        expected_keys = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volatility', 'momentum', 'stoch_k', 'stoch_d',
            'trend', 'rsi_signal', 'macd_signal_type', 'bb_signal'
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_signal_classifications(self, sample_ohlcv_data):
        df = add_all_indicators(sample_ohlcv_data)
        summary = get_indicator_summary(df, 100)
        assert summary['trend'] in ('bullish', 'bearish', 'neutral')
        assert summary['rsi_signal'] in ('overbought', 'oversold', 'neutral')
        assert summary['macd_signal_type'] in ('bullish', 'bearish', 'neutral')
        assert summary['bb_signal'] in ('above_upper', 'below_lower', 'neutral')

    def test_early_index_handles_nans(self, sample_ohlcv_data):
        df = add_all_indicators(sample_ohlcv_data)
        summary = get_indicator_summary(df, 0)
        # Should not raise; NaN values return as None
        assert summary['sma_50'] is None
