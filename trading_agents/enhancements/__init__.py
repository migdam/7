"""
Trading Agents Enhancements Module
Advanced features and tools for the trading framework
"""

# Core enhancements (no optional deps)
from trading_agents.enhancements.position_sizer import (
    PositionSizingEngine,
    FixedFractionalSizer,
    KellyCriterionSizer,
    ATRBasedSizer,
    VolatilityScaledSizer,
    OptimalFSizer,
    calculate_optimal_position
)
from trading_agents.enhancements.portfolio_analytics import (
    PortfolioAnalytics,
    create_equity_chart_ascii
)
from trading_agents.enhancements.sentiment_analyzer import (
    LexiconBasedAnalyzer,
    AggregatedSentimentAnalyzer,
    analyze_sentiment,
    analyze_news_batch
)
from trading_agents.enhancements.trade_analyzer import TradeAnalyzer, analyze_trades
from trading_agents.enhancements.strategy_optimizer import (
    StrategyOptimizer,
    RandomSearchOptimizer
)
from trading_agents.enhancements.monte_carlo import MonteCarloSimulator

# Optional: data fetcher (requires yfinance, requests)
try:
    from trading_agents.enhancements.data_fetcher import (
        YahooFinanceFetcher,
        AlphaVantageFetcher,
        CSVDataFetcher,
        get_data
    )
except ImportError:
    pass

# Optional: news fetcher (requires feedparser, requests)
try:
    from trading_agents.enhancements.news_fetcher import (
        RSSNewsFetcher,
        NewsAPIFetcher,
        FinancialModelingPrepFetcher,
        MockNewsFetcher,
        get_news
    )
except ImportError:
    pass

# Optional: VADER and TextBlob sentiment analyzers
try:
    from trading_agents.enhancements.sentiment_analyzer import VaderSentimentAnalyzer
except ImportError:
    pass

try:
    from trading_agents.enhancements.sentiment_analyzer import TextBlobAnalyzer
except ImportError:
    pass

# Optional: risk metrics (requires scipy)
try:
    from trading_agents.enhancements.risk_metrics import RiskMetrics
except ImportError:
    pass

# Optional: monitoring dashboard
try:
    from trading_agents.enhancements.monitoring_dashboard import TradingDashboard
except ImportError:
    pass

__all__ = [
    # Position sizing
    'PositionSizingEngine',
    'FixedFractionalSizer',
    'KellyCriterionSizer',
    'ATRBasedSizer',
    'VolatilityScaledSizer',
    'OptimalFSizer',
    'calculate_optimal_position',
    # Portfolio analytics
    'PortfolioAnalytics',
    'create_equity_chart_ascii',
    # Sentiment analysis
    'LexiconBasedAnalyzer',
    'AggregatedSentimentAnalyzer',
    'analyze_sentiment',
    'analyze_news_batch',
    # Trade analysis
    'TradeAnalyzer',
    'analyze_trades',
    # Strategy optimization
    'StrategyOptimizer',
    'RandomSearchOptimizer',
    # Monte Carlo
    'MonteCarloSimulator',
    # Optional (may not be available)
    'YahooFinanceFetcher',
    'AlphaVantageFetcher',
    'CSVDataFetcher',
    'get_data',
    'RSSNewsFetcher',
    'NewsAPIFetcher',
    'FinancialModelingPrepFetcher',
    'MockNewsFetcher',
    'get_news',
    'VaderSentimentAnalyzer',
    'TextBlobAnalyzer',
    'RiskMetrics',
    'TradingDashboard',
]
