# Trading Agents Enhancements

Advanced features and tools to supercharge your trading framework.

## 🚀 Overview

This module provides 10 powerful enhancements to extend the base trading framework:

1. **Market Data Fetcher** - Fetch real market data from multiple sources
2. **Advanced Position Sizing** - Sophisticated position sizing strategies
3. **Portfolio Analytics** - Comprehensive performance analysis
4. **News Fetcher** - Real-time news from RSS feeds and APIs
5. **Sentiment Analyzer** - AI-powered text sentiment analysis
6. **Risk Metrics** - Advanced risk calculations (VaR, CVaR, Beta, Alpha)
7. **Trade Analyzer** - Detailed trade performance analysis
8. **Strategy Optimizer** - Parameter optimization via grid/random search
9. **Monitoring Dashboard** - Real-time CLI dashboard
10. **Monte Carlo Simulator** - Scenario testing and robustness analysis

## 📦 Installation

### Core Enhancements (Included)
```bash
pip install scipy
```

### Optional Enhancements
```bash
# For market data fetching
pip install yfinance alpha-vantage

# For news fetching
pip install feedparser

# For sentiment analysis
pip install vaderSentiment textblob

# For visualization
pip install matplotlib plotly
```

## 1. Market Data Fetcher

Fetch real market data from Yahoo Finance, Alpha Vantage, or CSV files.

```python
from trading_agents.enhancements import get_data

# Fetch from Yahoo Finance
data = get_data('AAPL', start_date='2024-01-01', end_date='2024-12-31', source='yahoo')

# Fetch from Alpha Vantage
data = get_data('TSLA', source='alphavantage', api_key='your_key')

# Fetch from CSV
data = get_data('BTC', source='csv', file_path='data/bitcoin.csv')
```

## 2. Advanced Position Sizing

Multiple position sizing strategies for optimal capital allocation.

```python
from trading_agents.enhancements import PositionSizingEngine

# Kelly Criterion
sizer = PositionSizingEngine('kelly', fraction=0.5)  # Half Kelly
result = sizer.calculate(capital=100000, price=150.50, win_rate=0.55, avg_win=100, avg_loss=50)

# ATR-based sizing
sizer = PositionSizingEngine('atr', risk_per_atr=2.0, max_risk_pct=0.02)
result = sizer.calculate(capital=100000, price=150.50, atr=3.5)

# Volatility-scaled
sizer = PositionSizingEngine('volatility', target_volatility=0.15)
result = sizer.calculate(capital=100000, price=150.50, volatility=0.25)
```

**Available Strategies:**
- `fixed_fractional` - Fixed % of capital
- `kelly` - Kelly Criterion
- `atr` - ATR-based volatility adjustment
- `volatility` - Inverse volatility scaling
- `optimal_f` - Optimal F (Ralph Vince)

## 3. Portfolio Analytics

Comprehensive portfolio analysis with 15+ metrics.

```python
from trading_agents.enhancements import PortfolioAnalytics

analytics = PortfolioAnalytics(equity_curve, trades, risk_free_rate=0.02)

# Get all metrics
metrics = analytics.get_all_metrics()

# Print formatted report
analytics.print_report()

# Calculate specific metrics
sharpe = analytics.sharpe_ratio()
max_dd = analytics.max_drawdown()
calmar = analytics.calmar_ratio()
```

**Metrics Included:**
- Total Return & CAGR
- Sharpe, Sortino, Calmar Ratios
- Max Drawdown & Duration
- Win Rate & Profit Factor
- Ulcer Index
- Recovery Factor
- Consecutive wins/losses

## 4. News Fetcher

Fetch news from multiple sources for fundamental analysis.

```python
from trading_agents.enhancements import get_news

# RSS feeds
news = get_news(symbol='AAPL', source='rss', limit=10)

# NewsAPI
news = get_news(symbol='TSLA', source='newsapi', api_key='your_key', limit=20)

# Financial Modeling Prep
news = get_news(symbol='MSFT', source='fmp', api_key='your_key')

# Mock news (for testing)
news = get_news(symbol='GOOGL', source='mock')
```

## 5. Sentiment Analyzer

Analyze sentiment of news, social media, and text.

```python
from trading_agents.enhancements import analyze_sentiment, analyze_news_batch

# Analyze single text
result = analyze_sentiment("Stock surges on strong earnings beat", method='lexicon')
# {'sentiment': 'positive', 'score': 0.75}

# Use VADER (more accurate)
result = analyze_sentiment(text, method='vader', detailed=True)

# Analyze batch of news
news_items = [{'headline': 'Company posts record profits'}, ...]
batch_result = analyze_news_batch(news_items)
```

**Methods:**
- `lexicon` - Fast, no dependencies, finance-focused
- `vader` - VADER sentiment (requires vaderSentiment)
- `textblob` - TextBlob (requires textblob)
- `aggregate` - Combines all available methods

## 6. Risk Metrics

Advanced risk calculations for professional analysis.

```python
from trading_agents.enhancements import RiskMetrics

risk = RiskMetrics(returns, benchmark_returns, risk_free_rate=0.02)

# Value at Risk
var_95 = risk.value_at_risk(0.95, method='historical')
cvar_95 = risk.conditional_var(0.95)

# Benchmark-relative metrics
beta = risk.beta()
alpha = risk.alpha()
tracking_error = risk.tracking_error()
info_ratio = risk.information_ratio()

# Downside risk
downside_dev = risk.downside_deviation()
omega = risk.omega_ratio()
tail_ratio = risk.tail_ratio()

# Full report
risk.print_report()
```

## 7. Trade Analyzer

Detailed analysis of trading patterns and performance.

```python
from trading_agents.enhancements import TradeAnalyzer

analyzer = TradeAnalyzer(trades)

# Get comprehensive stats
stats = analyzer.get_trade_stats()

# Holding period analysis
holding_stats = analyzer.get_holding_period_stats()

# Time-based analysis
time_analysis = analyzer.get_time_analysis()  # Best hours/days

# Print full report
analyzer.print_report()
```

## 8. Strategy Optimizer

Optimize strategy parameters using grid or random search.

```python
from trading_agents.enhancements import StrategyOptimizer

def backtest(ma_period, rsi_threshold, position_size):
    # Your backtest logic
    return {'sharpe_ratio': 1.5, 'total_return': 15.2}

# Grid search
param_grid = {
    'ma_period': [10, 20, 30, 50],
    'rsi_threshold': [60, 70, 80],
    'position_size': [3, 5, 7]
}

optimizer = StrategyOptimizer(
    backtest_function=backtest,
    parameter_grid=param_grid,
    metric='sharpe_ratio',
    maximize=True
)

results = optimizer.grid_search(n_jobs=4, verbose=True)
print(f"Best parameters: {results['best_params']}")
```

**Features:**
- Grid search (exhaustive)
- Random search (faster for large spaces)
- Parallel execution
- Results DataFrame export
- Parameter impact visualization

## 9. Monitoring Dashboard

Real-time CLI dashboard for live monitoring.

```python
from trading_agents.enhancements import TradingDashboard

dashboard = TradingDashboard(update_interval=2.0)

# Define data source
def get_live_data():
    return {
        'portfolio': {...},
        'decision': {...},
        'metrics': {...},
        'trades': [...],
        'alerts': [...]
    }

# Monitor live (Ctrl+C to stop)
dashboard.monitor_live(get_live_data, duration=None)

# Or render once
dashboard.render_full_dashboard(
    portfolio_state=portfolio,
    decision_data=decision,
    metrics=metrics,
    trades=trades,
    alerts=alerts
)
```

## 10. Monte Carlo Simulator

Test strategy robustness across thousands of scenarios.

```python
from trading_agents.enhancements import MonteCarloSimulator

def backtest_function(data):
    # Run backtest on simulated data
    return {'sharpe_ratio': 1.5, 'total_return': 12.3}

mc = MonteCarloSimulator(
    backtest_function=backtest_function,
    base_data=historical_data,
    n_simulations=1000,
    random_seed=42
)

# Run simulation
results = mc.run_simulation(method='bootstrap', parallel=True, n_jobs=4)

# Print summary
mc.print_summary()

# Get confidence intervals
ci = mc.get_confidence_intervals('total_return_pct', confidence=0.95)
print(f"95% CI: {ci['lower_bound']:.2f}% to {ci['upper_bound']:.2f}%")
```

**Methods:**
- `bootstrap` - Resample historical returns
- `gbm` - Geometric Brownian Motion

## 🔗 Integration with Main Framework

All enhancements seamlessly integrate with the main trading framework:

```python
from trading_agents import TradingFirm
from trading_agents.enhancements import (
    get_data,
    PositionSizingEngine,
    get_news,
    analyze_sentiment
)

# Fetch real data
data = get_data('AAPL', start_date='2024-01-01', end_date='2024-12-31')

# Get news for context
news = get_news('AAPL', source='mock', limit=10)

# Initialize trading firm
firm = TradingFirm(llm_provider="openai", initial_capital=100000)

# Run backtest
firm.run_backtest(data=data, news_data=news)

# Analyze results with enhancements
from trading_agents.enhancements import PortfolioAnalytics, RiskMetrics

results = firm.get_results()
analytics = PortfolioAnalytics(results['equity_curve'], results['trades'])
analytics.print_report()

risk = RiskMetrics(results['equity_curve'].pct_change())
risk.print_report()
```

## 📊 Complete Example

```python
from trading_agents import TradingFirm
from trading_agents.enhancements import *

# 1. Fetch real market data
print("Fetching market data...")
data = get_data('SPY', start_date='2024-01-01', end_date='2024-12-31')

# 2. Get news and sentiment
print("Fetching news...")
news = get_news('SPY', source='mock', limit=20)
sentiment = analyze_news_batch(news)
print(f"News sentiment: {sentiment['overall_sentiment']}")

# 3. Run base backtest
print("Running backtest...")
firm = TradingFirm(llm_provider="openai", initial_capital=100000)
firm.run_backtest(data=data, news_data=news)

# 4. Advanced analytics
results = firm.get_results()
analytics = PortfolioAnalytics(results['equity_curve'], results['trades'])
analytics.print_report()

# 5. Risk analysis
risk = RiskMetrics(results['equity_curve'].pct_change())
risk.print_report()

# 6. Trade analysis
trade_analyzer = TradeAnalyzer(results['trades'])
trade_analyzer.print_report()

# 7. Monte Carlo robustness test
print("Running Monte Carlo simulation...")
mc = MonteCarloSimulator(
    backtest_function=lambda d: firm.run_backtest(d),
    base_data=data,
    n_simulations=100
)
mc.run_simulation()
mc.print_summary()

print("Complete analysis finished!")
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic enhancements, add complexity as needed
2. **Validate Data**: Always verify fetched data quality
3. **Use Appropriate Methods**: Choose position sizing strategy based on your risk tolerance
4. **Monitor Performance**: Use dashboard for live trading, analytics for backtests
5. **Test Robustness**: Run Monte Carlo simulations before live deployment
6. **Optimize Carefully**: Avoid overfitting when optimizing parameters

## 🔧 Troubleshooting

**ImportError for optional dependencies:**
```bash
pip install yfinance feedparser vaderSentiment textblob matplotlib
```

**API key errors:**
- Set environment variables: `NEWSAPI_KEY`, `ALPHAVANTAGE_KEY`
- Or pass directly: `get_news(api_key='your_key')`

**Slow optimization:**
- Use `n_jobs` parameter for parallel execution
- Consider random search instead of grid search
- Reduce parameter grid size

## 📚 Learn More

- Full documentation: See main [README.md](../../README.md)
- Examples: Check `examples/` directory for advanced use cases
- PRD: See original Product Requirements Document for architecture details

---

**These enhancements transform the framework from MVP to production-ready professional trading system! 🚀**
