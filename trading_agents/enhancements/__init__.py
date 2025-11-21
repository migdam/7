"""
Trading Agents Enhancements Module
Advanced features and tools for the trading framework
"""

from trading_agents.enhancements.data_fetcher import (
    YahooFinanceFetcher,
    AlphaVantageFetcher,
    CSVDataFetcher,
    get_data
)
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
from trading_agents.enhancements.news_fetcher import (
    RSSNewsFetcher,
    NewsAPIFetcher,
    FinancialModelingPrepFetcher,
    MockNewsFetcher,
    get_news
)
from trading_agents.enhancements.sentiment_analyzer import (
    LexiconBasedAnalyzer,
    VaderSentimentAnalyzer,
    TextBlobAnalyzer,
    AggregatedSentimentAnalyzer,
    analyze_sentiment,
    analyze_news_batch
)
from trading_agents.enhancements.risk_metrics import RiskMetrics
from trading_agents.enhancements.trade_analyzer import TradeAnalyzer, analyze_trades
from trading_agents.enhancements.strategy_optimizer import (
    StrategyOptimizer,
    RandomSearchOptimizer
)
from trading_agents.enhancements.monitoring_dashboard import TradingDashboard
from trading_agents.enhancements.monte_carlo import MonteCarloSimulator

__all__ = [
    # Data fetching
    'YahooFinanceFetcher',
    'AlphaVantageFetcher',
    'CSVDataFetcher',
    'get_data',
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
    # News fetching
    'RSSNewsFetcher',
    'NewsAPIFetcher',
    'FinancialModelingPrepFetcher',
    'MockNewsFetcher',
    'get_news',
    # Sentiment analysis
    'LexiconBasedAnalyzer',
    'VaderSentimentAnalyzer',
    'TextBlobAnalyzer',
    'AggregatedSentimentAnalyzer',
    'analyze_sentiment',
    'analyze_news_batch',
    # Risk metrics
    'RiskMetrics',
    # Trade analysis
    'TradeAnalyzer',
    'analyze_trades',
    # Strategy optimization
    'StrategyOptimizer',
    'RandomSearchOptimizer',
    # Monitoring
    'TradingDashboard',
    # Monte Carlo
    'MonteCarloSimulator'
]
