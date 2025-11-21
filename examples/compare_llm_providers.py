"""
Compare LLM Providers Example
Test different LLM providers (OpenAI, Anthropic, Ollama) on same data
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from trading_agents.main import TradingFirm


def generate_simple_data(days: int = 100) -> pd.DataFrame:
    """Generate simple test data"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

    price = 100.0
    prices = []
    for i in range(days):
        price += np.random.normal(0.3, 2.0)
        prices.append(max(price, 1.0))

    data = pd.DataFrame({
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)

    return data


def test_provider(provider: str, model: str = None, data: pd.DataFrame = None):
    """
    Test a single LLM provider

    Args:
        provider: Provider name
        model: Model name (optional)
        data: Market data

    Returns:
        Metrics dictionary
    """
    print(f"\nTesting {provider.upper()}" + (f" ({model})" if model else ""))
    print("-" * 50)

    try:
        firm = TradingFirm(
            llm_provider=provider,
            llm_model=model,
            initial_capital=100000.0,
            max_position_size=3.0,
            log_dir="logs",
            verbose=False
        )

        metrics = firm.run_backtest(
            data=data,
            initial_capital=100000.0
        )

        print(f"✓ {provider.upper()} completed successfully")
        return metrics

    except Exception as e:
        print(f"✗ {provider.upper()} failed: {str(e)}")
        return None


def main():
    """Compare different LLM providers"""

    print("=" * 70)
    print("COMPARE LLM PROVIDERS")
    print("=" * 70)

    print("\nGenerating test data...")
    data = generate_simple_data(days=100)
    print(f"Generated {len(data)} days of data")

    # Test configurations
    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        # ("ollama", "llama3.1"),  # Uncomment if you have Ollama running
    ]

    results = {}

    print("\nTesting providers...")
    print("This may take several minutes depending on API response times")

    for provider, model in providers_to_test:
        metrics = test_provider(provider, model, data)
        if metrics:
            results[f"{provider}_{model}"] = metrics

    # Display comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        # Header
        providers_list = list(results.keys())
        header = "{:<30}".format("Metric")
        for provider in providers_list:
            header += " {:>18}".format(provider.split('_')[0].upper())
        print(header)
        print("-" * 70)

        # Metrics
        metrics_to_show = [
            ("Total Return %", "total_return_pct"),
            ("Max Drawdown %", "max_drawdown_pct"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Win Rate %", "win_rate"),
            ("Total Trades", "total_trades"),
        ]

        for label, key in metrics_to_show:
            row = "{:<30}".format(label)
            for provider in providers_list:
                val = results[provider].get(key, 0)
                if key == "win_rate":
                    val *= 100
                if "%" in label or key == "win_rate":
                    row += " {:>18.2f}".format(val)
                elif key == "total_trades":
                    row += " {:>18.0f}".format(val)
                else:
                    row += " {:>18.2f}".format(val)
            print(row)

        print("\n" + "=" * 70)

        # Analysis
        print("\nKey Observations:")
        print("- Different LLMs may produce different trading strategies")
        print("- More expensive models (GPT-4, Claude Opus) may be more cautious")
        print("- Cheaper models (GPT-3.5, Claude Haiku) may be more aggressive")
        print("- Local models (Ollama) offer privacy and cost savings")
        print("- Consider speed vs. quality tradeoffs for your use case")

    elif len(results) == 1:
        print("\n" + "=" * 70)
        print("SINGLE PROVIDER TEST RESULTS")
        print("=" * 70)

        provider = list(results.keys())[0]
        metrics = results[provider]

        print(f"\nProvider: {provider}")
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")

    else:
        print("\n✗ No providers successfully completed the test")
        print("\nTroubleshooting:")
        print("1. Check that API keys are set in .env file")
        print("2. Verify internet connection for API providers")
        print("3. Ensure Ollama is running (if testing local models)")
        print("4. Check that all dependencies are installed")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
