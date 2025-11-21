"""
Main Trading System Orchestrator
Coordinates all agents in the virtual trading firm
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm

from trading_agents.core.llm import get_llm
from trading_agents.core.market_env import MarketEnvironment
from trading_agents.data.indicators import add_all_indicators, get_indicator_summary
from trading_agents.agents import (
    MarketAnalystAgent,
    NewsAnalystAgent,
    StrategyAgent,
    CIOAgent,
    RiskManagerAgent,
    ExecutionAgent,
    TradeLoggerAgent
)


class TradingFirm:
    """
    Virtual Trading Firm
    Orchestrates multi-agent decision-making process
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        initial_capital: float = 100000.0,
        max_position_size: float = 10.0,
        max_drawdown_pct: float = 0.15,
        log_dir: str = "logs",
        verbose: bool = True
    ):
        """
        Initialize Trading Firm

        Args:
            llm_provider: LLM provider ('openai', 'anthropic', 'ollama')
            llm_model: Specific model to use
            initial_capital: Starting capital
            max_position_size: Maximum position size
            max_drawdown_pct: Maximum drawdown threshold
            log_dir: Directory for logs
            verbose: Print progress
        """
        self.verbose = verbose

        # Initialize LLM
        if self.verbose:
            print(f"Initializing LLM: {llm_provider} ({llm_model or 'default'})")

        self.llm = get_llm(
            provider_name=llm_provider,
            model=llm_model,
            temperature=0.5,
            max_tokens=1000
        )

        # Initialize agents
        if self.verbose:
            print("Initializing agents...")

        self.market_analyst = MarketAnalystAgent(self.llm)
        self.news_analyst = NewsAnalystAgent(self.llm)
        self.strategy_analyst = StrategyAgent(self.llm)
        self.cio = CIOAgent(self.llm)
        self.risk_manager = RiskManagerAgent(
            self.llm,
            max_position_size=max_position_size,
            max_drawdown_pct=max_drawdown_pct
        )
        self.execution_trader = ExecutionAgent(self.llm)
        self.logger = TradeLoggerAgent(log_dir=log_dir)

        # Environment (will be set during run)
        self.env: Optional[MarketEnvironment] = None
        self.data_with_indicators: Optional[pd.DataFrame] = None

        if self.verbose:
            print("Trading Firm initialized successfully!")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with indicators

        Args:
            data: Raw OHLCV DataFrame

        Returns:
            DataFrame with indicators
        """
        if self.verbose:
            print("Calculating technical indicators...")

        return add_all_indicators(data)

    def run_backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        news_data: Optional[list] = None,
        events_data: Optional[list] = None,
        start_idx: int = 50  # Skip first N bars for indicator warmup
    ):
        """
        Run backtest simulation

        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital
            news_data: Optional news data
            events_data: Optional events data
            start_idx: Starting index (for indicator warmup)
        """
        # Prepare data with indicators
        self.data_with_indicators = self.prepare_data(data)

        # Initialize environment
        self.env = MarketEnvironment(
            data=self.data_with_indicators,
            initial_capital=initial_capital,
            max_position_size=self.risk_manager.max_position_size
        )

        # Skip to start index
        self.env.current_idx = start_idx

        if self.verbose:
            print(f"\nStarting backtest from index {start_idx} to {len(data)-1}")
            print(f"Initial capital: ${initial_capital:,.2f}")
            print("="*60)

        # Main trading loop
        pbar = tqdm(total=len(data)-start_idx, disable=not self.verbose)

        while self.env.step():
            # Run decision cycle
            self._decision_cycle(news_data, events_data)
            pbar.update(1)

        pbar.close()

        # Print final results
        if self.verbose:
            print("\n" + "="*60)
            print("BACKTEST COMPLETED")
            print("="*60)

            metrics = self.env.get_performance_metrics()
            print(f"Final Equity: ${metrics['final_equity']:,.2f}")
            print(f"Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']*100:.1f}%")

            # Logger summary
            self.logger.print_summary()

        # Save logs
        self.logger.save_logs(prefix="backtest")

        return metrics

    def _decision_cycle(
        self,
        news_data: Optional[list] = None,
        events_data: Optional[list] = None
    ):
        """
        Execute one complete decision cycle

        Args:
            news_data: News items
            events_data: Events data
        """
        # Get current state
        current_time = self.env.current_time
        current_price = self.env.current_price
        current_position = self.env.get_position_info()
        window_data = self.env.get_window(lookback=20)

        # Get indicators for current bar
        indicators = get_indicator_summary(self.data_with_indicators, self.env.current_idx)

        # Prepare price history
        price_history = []
        for idx in range(max(0, self.env.current_idx - 10), self.env.current_idx + 1):
            bar = self.data_with_indicators.iloc[idx]
            price_history.append({
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close']
            })

        # === AGENT DECISION PIPELINE ===

        # 1. Market Analyst
        market_context = {
            'current_price': current_price,
            'indicators': indicators,
            'price_history': price_history
        }
        market_analysis = self.market_analyst.process(market_context)

        # 2. News Analyst
        news_context = {
            'news': news_data or [],
            'events': events_data or [],
            'current_date': current_time
        }
        news_analysis = self.news_analyst.process(news_context)

        # 3. Strategy Analyst
        strategy_context = {
            'market_analysis': market_analysis,
            'news_analysis': news_analysis,
            'current_position': current_position,
            'current_price': current_price
        }
        strategy = self.strategy_analyst.process(strategy_context)

        # 4. CIO Decision
        cio_context = {
            'market_analysis': market_analysis,
            'news_analysis': news_analysis,
            'strategy': strategy,
            'current_position': current_position,
            'current_price': current_price
        }
        cio_decision = self.cio.process(cio_context)

        # 5. Risk Manager
        risk_context = {
            'decision': cio_decision,
            'portfolio': {
                'equity': self.env.get_equity(),
                'cash': self.env.cash,
                'initial_capital': self.env.initial_capital,
                'current_price': current_price
            },
            'current_position': current_position,
            'current_price': current_price
        }
        risk_approval = self.risk_manager.process(risk_context)

        # 6. Execution Trader
        execution_context = {
            'decision': cio_decision,
            'risk_approval': risk_approval,
            'current_price': current_price,
            'current_position': current_position
        }
        execution_order = self.execution_trader.process(execution_context, use_llm=False)

        # 7. Execute Trade in Environment
        trade_result = None
        if execution_order['action'] in ['BUY', 'SELL']:
            trade_result = self.env.execute_trade(
                size=execution_order['size'],
                rationale=cio_decision.get('justification', '')
            )
        elif execution_order['action'] == 'CLOSE':
            trade_result = self.env.execute_trade(
                size=0,
                rationale=cio_decision.get('justification', '')
            )

        # Log trade if executed
        if trade_result and trade_result.get('success'):
            trade_obj = trade_result.get('trade')
            if trade_obj:
                self.logger.log_trade(
                    timestamp=current_time,
                    action=execution_order['action'],
                    size=execution_order['size'],
                    price=current_price,
                    pnl=getattr(trade_obj, 'pnl', 0),
                    rationale=cio_decision.get('justification', '')
                )

        # 8. Logger - Log Complete Cycle
        log_context = {
            'timestamp': current_time,
            'current_price': current_price,
            'market_analysis': market_analysis,
            'news_analysis': news_analysis,
            'strategy': strategy,
            'cio_decision': cio_decision,
            'risk_approval': risk_approval,
            'execution_order': execution_order,
            'portfolio_state': {
                'equity': self.env.get_equity(),
                'cash': self.env.cash,
                'position': current_position
            }
        }
        self.logger.process(log_context)

    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results

        Returns:
            Dictionary with results and metrics
        """
        if not self.env:
            return {}

        metrics = self.env.get_performance_metrics()
        equity_df = self.env.get_equity_dataframe()
        logger_summary = self.logger.get_summary()

        return {
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': self.env.trades,
            'logger_summary': logger_summary
        }


def main():
    """Example usage"""
    # Generate sample data
    print("Generating sample data...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # Simulate price data with trend
    price = 100.0
    prices = []
    for i in range(len(dates)):
        price += np.random.normal(0.5, 2)  # Upward drift with noise
        prices.append(price)

    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }, index=dates)

    print(f"Generated {len(data)} days of data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Initialize trading firm
    firm = TradingFirm(
        llm_provider="openai",  # Change to "anthropic" or "ollama" as needed
        initial_capital=100000.0,
        max_position_size=5.0,
        verbose=True
    )

    # Run backtest
    firm.run_backtest(
        data=data,
        initial_capital=100000.0
    )

    # Get results
    results = firm.get_results()

    print("\nBacktest complete! Check logs/ directory for detailed results.")


if __name__ == "__main__":
    main()
