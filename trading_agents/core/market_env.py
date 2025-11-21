"""
Market Environment Simulator
Pandas-based simulation environment for backtesting trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return self.size * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rationale: str = ""


class MarketEnvironment:
    """
    Trading environment simulator
    Manages positions, executes trades, tracks equity
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_position_size: float = 10.0,
        commission: float = 0.001,  # 0.1% per trade
        symbol: str = "ASSET"
    ):
        """
        Initialize market environment

        Args:
            data: DataFrame with OHLCV data (index: datetime, columns: open, high, low, close, volume)
            initial_capital: Starting capital
            max_position_size: Maximum position size (units)
            commission: Commission rate (fraction)
            symbol: Trading symbol name
        """
        self.data = data.copy()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission

        # State variables
        self.current_idx = 0
        self.cash = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def reset(self):
        """Reset environment to initial state"""
        self.current_idx = 0
        self.cash = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def step(self) -> bool:
        """
        Move to next time step

        Returns:
            True if more data available, False if reached end
        """
        if self.current_idx >= len(self.data) - 1:
            return False

        self.current_idx += 1

        # Update position with current price
        if self.position:
            self.position.current_price = self.current_price

        # Track equity
        equity = self.get_equity()
        self.equity_curve.append(equity)
        self.timestamps.append(self.current_time)

        return True

    @property
    def current_time(self) -> datetime:
        """Current timestamp"""
        return self.data.index[self.current_idx]

    @property
    def current_price(self) -> float:
        """Current close price"""
        return self.data.iloc[self.current_idx]['close']

    @property
    def current_bar(self) -> pd.Series:
        """Current OHLCV bar"""
        return self.data.iloc[self.current_idx]

    def get_window(self, lookback: int = 20) -> pd.DataFrame:
        """
        Get historical window of data

        Args:
            lookback: Number of bars to look back

        Returns:
            DataFrame with historical data
        """
        start_idx = max(0, self.current_idx - lookback + 1)
        return self.data.iloc[start_idx:self.current_idx + 1]

    def get_equity(self) -> float:
        """
        Calculate total equity (cash + position value)

        Returns:
            Total equity value
        """
        equity = self.cash
        if self.position:
            equity += self.position.market_value
        return equity

    def get_position_info(self) -> Dict:
        """Get current position information"""
        if not self.position:
            return {
                "has_position": False,
                "size": 0.0,
                "market_value": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0
            }

        return {
            "has_position": True,
            "size": self.position.size,
            "entry_price": self.position.entry_price,
            "current_price": self.position.current_price,
            "market_value": self.position.market_value,
            "unrealized_pnl": self.position.unrealized_pnl,
            "unrealized_pnl_pct": self.position.unrealized_pnl_pct
        }

    def can_trade(self, size: float) -> Tuple[bool, str]:
        """
        Check if trade is valid

        Args:
            size: Position size (positive for long, negative for short)

        Returns:
            (can_trade, reason)
        """
        # Check position size limit
        if abs(size) > self.max_position_size:
            return False, f"Size {size} exceeds max position size {self.max_position_size}"

        # Check if we have an existing position
        if self.position and size != 0:
            # Can only close or modify existing position
            if (self.position.size > 0 and size > 0) or (self.position.size < 0 and size < 0):
                return False, "Already have position in same direction"

        # Check capital requirements for new position
        if not self.position and size != 0:
            required_capital = abs(size) * self.current_price * (1 + self.commission)
            if required_capital > self.cash:
                return False, f"Insufficient capital: need {required_capital:.2f}, have {self.cash:.2f}"

        return True, "OK"

    def execute_trade(self, size: float, rationale: str = "") -> Dict:
        """
        Execute a trade

        Args:
            size: Position size (positive=long, negative=short, 0=close)
            rationale: Reason for trade

        Returns:
            Trade result dictionary
        """
        can_trade, reason = self.can_trade(size)
        if not can_trade:
            return {
                "success": False,
                "reason": reason,
                "trade": None
            }

        price = self.current_price
        timestamp = self.current_time
        commission_paid = abs(size) * price * self.commission

        # Closing existing position
        if self.position and size == 0:
            pnl = self.position.unrealized_pnl - commission_paid
            pnl_pct = self.position.unrealized_pnl_pct

            trade = Trade(
                symbol=self.symbol,
                side="SELL" if self.position.size > 0 else "BUY",
                size=abs(self.position.size),
                price=price,
                timestamp=timestamp,
                pnl=pnl,
                pnl_pct=pnl_pct,
                rationale=rationale
            )

            self.cash += self.position.size * price - commission_paid
            self.position = None
            self.trades.append(trade)
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            return {
                "success": True,
                "action": "CLOSE",
                "trade": trade,
                "equity": self.get_equity()
            }

        # Opening new position
        elif not self.position and size != 0:
            self.cash -= abs(size) * price + commission_paid

            self.position = Position(
                symbol=self.symbol,
                size=size,
                entry_price=price,
                entry_time=timestamp,
                current_price=price
            )

            trade = Trade(
                symbol=self.symbol,
                side="BUY" if size > 0 else "SELL",
                size=abs(size),
                price=price,
                timestamp=timestamp,
                rationale=rationale
            )

            return {
                "success": True,
                "action": "OPEN",
                "trade": trade,
                "equity": self.get_equity()
            }

        return {
            "success": False,
            "reason": "No valid action",
            "trade": None
        }

    def get_state(self) -> Dict:
        """
        Get complete environment state

        Returns:
            Dictionary with current state
        """
        return {
            "timestamp": self.current_time,
            "price": self.current_price,
            "cash": self.cash,
            "equity": self.get_equity(),
            "position": self.get_position_info(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        }

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics

        Returns:
            Dictionary with performance metrics
        """
        if len(self.equity_curve) < 2:
            return {}

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio (annualized, assuming daily data)
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

        # Sortino ratio (downside deviation)
        sortino_ratio = 0.0
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()

        # Win rate and profit factor
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        winning_pnl = sum(t.pnl for t in self.trades if t.pnl > 0)
        losing_pnl = sum(abs(t.pnl) for t in self.trades if t.pnl < 0)
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0

        return {
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "final_equity": equity_series.iloc[-1]
        }

    def get_equity_dataframe(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame

        Returns:
            DataFrame with timestamp and equity columns
        """
        if len(self.timestamps) == 0:
            return pd.DataFrame()

        return pd.DataFrame({
            "timestamp": self.timestamps,
            "equity": self.equity_curve[1:]  # Skip initial value
        }).set_index("timestamp")
