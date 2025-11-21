"""
Advanced Backtest Example with News
Demonstrates using news/events data in the trading system
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_agents.main import TradingFirm


def generate_market_data_with_events(
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    initial_price: float = 100.0
) -> pd.DataFrame:
    """
    Generate market data with simulated event impacts

    Args:
        start_date: Start date
        end_date: End date
        initial_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate base prices
    price = initial_price
    prices = []

    # Simulate some major events
    event_dates = {
        pd.Timestamp('2024-03-15'): 5.0,   # Positive event
        pd.Timestamp('2024-06-20'): -8.0,  # Negative event
        pd.Timestamp('2024-09-10'): 6.0,   # Positive event
    }

    for date in dates:
        # Base movement
        daily_return = np.random.normal(0.1, 2.0)

        # Add event impact
        if date in event_dates:
            daily_return += event_dates[date]

        price += daily_return
        prices.append(max(price, 1.0))

    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)

    return data


def generate_news_feed():
    """Generate sample news items"""
    return [
        {
            'headline': 'Strong earnings report exceeds expectations',
            'source': 'Financial Times',
            'sentiment': 'positive'
        },
        {
            'headline': 'New product launch receives positive reviews',
            'source': 'TechCrunch',
            'sentiment': 'positive'
        },
        {
            'headline': 'Regulatory concerns raised by analysts',
            'source': 'Reuters',
            'sentiment': 'negative'
        },
        {
            'headline': 'Market outlook remains stable',
            'source': 'Bloomberg',
            'sentiment': 'neutral'
        }
    ]


def generate_events_calendar():
    """Generate sample scheduled events"""
    return [
        {
            'name': 'FOMC Meeting',
            'time': '2:00 PM EST',
            'importance': 'high'
        },
        {
            'name': 'Earnings Call',
            'time': '4:30 PM EST',
            'importance': 'high'
        },
        {
            'name': 'Economic Data Release',
            'time': '8:30 AM EST',
            'importance': 'medium'
        }
    ]


def run_comparison_backtests():
    """Run backtests with and without news to compare"""

    print("=" * 70)
    print("ADVANCED BACKTEST - WITH NEWS & EVENTS")
    print("=" * 70)

    # Generate data
    print("\nGenerating market data with event impacts...")
    data = generate_market_data_with_events()
    news_feed = generate_news_feed()
    events_calendar = generate_events_calendar()

    print(f"Generated {len(data)} days of data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # --- Backtest WITHOUT news ---
    print("\n" + "=" * 70)
    print("BACKTEST 1: WITHOUT NEWS DATA")
    print("=" * 70)

    firm1 = TradingFirm(
        llm_provider="openai",
        initial_capital=100000.0,
        max_position_size=5.0,
        log_dir="logs",
        verbose=False  # Quiet mode for comparison
    )

    print("\nRunning backtest without news...")
    metrics1 = firm1.run_backtest(
        data=data,
        initial_capital=100000.0,
        news_data=None,
        events_data=None
    )

    # --- Backtest WITH news ---
    print("\n" + "=" * 70)
    print("BACKTEST 2: WITH NEWS DATA")
    print("=" * 70)

    firm2 = TradingFirm(
        llm_provider="openai",
        initial_capital=100000.0,
        max_position_size=5.0,
        log_dir="logs",
        verbose=False
    )

    print("\nRunning backtest with news and events...")
    metrics2 = firm2.run_backtest(
        data=data,
        initial_capital=100000.0,
        news_data=news_feed,
        events_data=events_calendar
    )

    # --- Compare Results ---
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n{:<30} {:>18} {:>18}".format("Metric", "Without News", "With News"))
    print("-" * 70)

    metrics = [
        ("Final Equity", "final_equity", "${:,.2f}"),
        ("Total Return", "total_return_pct", "{:.2f}%"),
        ("Max Drawdown", "max_drawdown_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
    ]

    for label, key, fmt in metrics:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)

        # Special handling for percentages
        if key == "win_rate":
            val1 *= 100
            val2 *= 100

        formatted_val1 = fmt.format(val1)
        formatted_val2 = fmt.format(val2)

        # Add improvement indicator
        if key in ["final_equity", "total_return_pct", "sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor"]:
            if val2 > val1:
                indicator = " ✓"
            elif val2 < val1:
                indicator = " ✗"
            else:
                indicator = " ="
        elif key in ["max_drawdown_pct"]:
            if val2 > val1:  # More negative drawdown is worse
                indicator = " ✗"
            elif val2 < val1:
                indicator = " ✓"
            else:
                indicator = " ="
        else:
            indicator = ""

        print("{:<30} {:>18} {:>18}{}".format(label, formatted_val1, formatted_val2, indicator))

    print("\n" + "=" * 70)
    print("✓ = Better with news  |  ✗ = Worse with news  |  = No change")
    print("=" * 70)

    print("\nKey Insights:")
    print("- News data provides additional context for decision-making")
    print("- Event-driven volatility can be better managed with awareness")
    print("- Combining technical and fundamental analysis improves robustness")

    print("\nLogs saved to logs/ directory for detailed analysis")


def main():
    """Run advanced backtest"""
    try:
        run_comparison_backtests()
        print("\n" + "=" * 70)
        print("ADVANCED BACKTEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set your API keys in .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
