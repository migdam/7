"""
Technical Indicators Module
Calculate technical analysis indicators for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """MACD (Moving Average Convergence Divergence)"""
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    }


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """Bollinger Bands"""
    middle = calculate_sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    return {
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(com=period - 1, min_periods=period).mean()
    return atr


def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling annualized volatility"""
    returns = series.pct_change()
    return returns.rolling(window=period).std() * np.sqrt(252)


def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Price momentum (rate of change)"""
    return (series / series.shift(period) - 1) * 100


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> Dict[str, pd.Series]:
    """Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    denom = high_max - low_min
    stoch_k = ((df['close'] - low_min) / denom.where(denom != 0, np.nan)) * 100
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    }


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with indicator columns added
    """
    result = df.copy()
    close = result['close']

    # Moving averages
    result['sma_20'] = calculate_sma(close, 20)
    result['sma_50'] = calculate_sma(close, 50)
    result['ema_12'] = calculate_ema(close, 12)
    result['ema_26'] = calculate_ema(close, 26)

    # RSI
    result['rsi'] = calculate_rsi(close, 14)

    # MACD
    macd = calculate_macd(close)
    result['macd'] = macd['macd']
    result['macd_signal'] = macd['macd_signal']
    result['macd_histogram'] = macd['macd_histogram']

    # Bollinger Bands
    bb = calculate_bollinger_bands(close, 20)
    result['bb_upper'] = bb['bb_upper']
    result['bb_middle'] = bb['bb_middle']
    result['bb_lower'] = bb['bb_lower']

    # ATR
    result['atr'] = calculate_atr(result, 14)

    # Volatility
    result['volatility'] = calculate_volatility(close, 20)

    # Momentum
    result['momentum'] = calculate_momentum(close, 10)

    # Stochastic
    stoch = calculate_stochastic(result)
    result['stoch_k'] = stoch['stoch_k']
    result['stoch_d'] = stoch['stoch_d']

    return result


def get_indicator_summary(df: pd.DataFrame, idx: int) -> Dict:
    """
    Get indicator values and signal classifications for a specific bar.

    Args:
        df: DataFrame with indicator columns (from add_all_indicators)
        idx: Row index

    Returns:
        Dictionary with indicator values and signal classifications
    """
    row = df.iloc[idx]

    def safe_get(key: str, default=None):
        val = row.get(key, default)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return round(float(val), 4) if isinstance(val, (int, float, np.floating)) else val
        return default

    rsi = safe_get('rsi')
    macd_val = safe_get('macd')
    macd_sig = safe_get('macd_signal')
    sma_20 = safe_get('sma_20')
    sma_50 = safe_get('sma_50')
    bb_upper = safe_get('bb_upper')
    bb_lower = safe_get('bb_lower')
    close = safe_get('close')

    # Trend signal
    trend = 'neutral'
    if sma_20 is not None and sma_50 is not None:
        if sma_20 > sma_50:
            trend = 'bullish'
        elif sma_20 < sma_50:
            trend = 'bearish'

    # RSI signal
    rsi_signal = 'neutral'
    if rsi is not None:
        if rsi > 70:
            rsi_signal = 'overbought'
        elif rsi < 30:
            rsi_signal = 'oversold'

    # MACD signal type
    macd_signal_type = 'neutral'
    if macd_val is not None and macd_sig is not None:
        if macd_val > macd_sig:
            macd_signal_type = 'bullish'
        elif macd_val < macd_sig:
            macd_signal_type = 'bearish'

    # Bollinger Band signal
    bb_signal = 'neutral'
    if close is not None and bb_upper is not None and bb_lower is not None:
        if close > bb_upper:
            bb_signal = 'above_upper'
        elif close < bb_lower:
            bb_signal = 'below_lower'

    return {
        'rsi': rsi,
        'macd': macd_val,
        'macd_signal': macd_sig,
        'macd_histogram': safe_get('macd_histogram'),
        'sma_20': sma_20,
        'sma_50': sma_50,
        'bb_upper': bb_upper,
        'bb_middle': safe_get('bb_middle'),
        'bb_lower': bb_lower,
        'atr': safe_get('atr'),
        'volatility': safe_get('volatility'),
        'momentum': safe_get('momentum'),
        'stoch_k': safe_get('stoch_k'),
        'stoch_d': safe_get('stoch_d'),
        'trend': trend,
        'rsi_signal': rsi_signal,
        'macd_signal_type': macd_signal_type,
        'bb_signal': bb_signal
    }
