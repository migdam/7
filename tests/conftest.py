"""Shared test fixtures"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from trading_agents.core.llm import BaseLLM


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    price = 100.0
    prices = []
    for _ in range(252):
        price += np.random.normal(0.1, 1.5)
        prices.append(max(price, 1.0))

    df = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)
    return df


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns valid JSON responses."""
    llm = MagicMock(spec=BaseLLM)
    llm.generate.return_value = "Test response"
    llm.generate_json.return_value = {
        "decision": "HOLD",
        "confidence": 0.5,
        "reasoning": "Test"
    }
    return llm
