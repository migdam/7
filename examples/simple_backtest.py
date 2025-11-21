"""
Simple Backtest Example
Demonstrates basic usage of the Trading Agents framework
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from trading_agents.main import TradingFirm


def generate_sample_data(
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    initial_price: float = 100.0,
    drift: float = 0.5,
    volatility: float = 2.0
) -> pd.DataFrame:
    """
    Generate sample OHLCV data with random walk

    Args:
        start_date: Start date
        end_date: End date
        initial_price: Starting price
        drift: Daily drift (trend)
        volatility: Daily volatility

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate prices with trend
    price = initial_price
    prices = []
    for i in range(len(dates)):
        price += np.random.normal(drift, volatility)
        prices.append(max(price, 1.0))  # Ensure positive prices

    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)

    return data


def main():
    """Run simple backtest example"""

    print("=" * 60)
    print("TRADING AGENTS - SIMPLE BACKTEST EXAMPLE")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample market data...")
    data = generate_sample_data(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_price=100.0,
        drift=0.3,  # Slight upward trend
        volatility=2.0
    )

    print(f"   Generated {len(data)} days of data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Average volume: {data['volume'].mean():,.0f}")

    # Initialize trading firm
    print("\n2. Initializing Trading Firm...")
    print("   Note: Make sure you have set your API keys in .env file")

    firm = TradingFirm(
        llm_provider="openai",  # Change to "anthropic" or "ollama" as needed
        llm_model=None,  # Use default model
        initial_capital=100000.0,
        max_position_size=5.0,
        max_drawdown_pct=0.15,
        log_dir="logs",
        verbose=True
    )

    # Run backtest
    print("\n3. Running backtest...")
    print("   This may take a few minutes depending on LLM response time...")

    metrics = firm.run_backtest(
        data=data,
        initial_capital=100000.0
    )

    # Display results
    print("\n4. Results Summary")
    print("=" * 60)
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print("=" * 60)

    # Get full results
    results = firm.get_results()

    # Show some trades
    if results['trades']:
        print("\n5. Sample Trades (first 5):")
        for i, trade in enumerate(results['trades'][:5]):
            print(f"\n   Trade {i+1}:")
            print(f"   Time: {trade.timestamp}")
            print(f"   Side: {trade.side}")
            print(f"   Size: {trade.size}")
            print(f"   Price: ${trade.price:.2f}")
            print(f"   P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            print(f"   Rationale: {trade.rationale[:100]}...")

    print("\n6. Logs saved to logs/ directory")
    print("   Check the JSON and CSV files for detailed analysis")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
