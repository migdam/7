"""
Portfolio Analytics and Visualization
Advanced portfolio analysis, metrics, and visualization tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class PortfolioAnalytics:
    """
    Comprehensive portfolio analytics engine
    """

    def __init__(self, equity_curve: pd.Series, trades: List = None, risk_free_rate: float = 0.02):
        """
        Initialize portfolio analytics

        Args:
            equity_curve: Time series of portfolio equity
            trades: List of trade objects
            risk_free_rate: Risk-free rate for Sharpe calculation (annualized)
        """
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate

        # Calculate returns
        self.returns = equity_curve.pct_change().dropna()

    def total_return(self) -> float:
        """Calculate total return percentage"""
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

    def cagr(self) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(self.equity_curve) < 2:
            return 0.0

        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25

        if years == 0:
            return 0.0

        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]
        cagr = (total_return ** (1 / years) - 1) * 100

        return cagr

    def volatility(self, annualized: bool = True) -> float:
        """Calculate volatility"""
        vol = self.returns.std()
        if annualized:
            vol *= np.sqrt(252)  # Assume 252 trading days
        return vol * 100

    def sharpe_ratio(self, annualized: bool = True) -> float:
        """Calculate Sharpe Ratio"""
        if len(self.returns) == 0 or self.returns.std() == 0:
            return 0.0

        excess_returns = self.returns - (self.risk_free_rate / 252)
        sharpe = excess_returns.mean() / self.returns.std()

        if annualized:
            sharpe *= np.sqrt(252)

        return sharpe

    def sortino_ratio(self, annualized: bool = True, target_return: float = 0.0) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        if len(self.returns) == 0:
            return 0.0

        excess_returns = self.returns - (target_return / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std()

        if annualized:
            sortino *= np.sqrt(252)

        return sortino

    def max_drawdown(self) -> Dict:
        """Calculate maximum drawdown"""
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Find drawdown duration
        dd_start = self.equity_curve[:max_dd_date].idxmax()
        dd_end_series = self.equity_curve[max_dd_date:]
        recovery_idx = dd_end_series[dd_end_series >= self.equity_curve[dd_start]].index
        dd_end = recovery_idx[0] if len(recovery_idx) > 0 else self.equity_curve.index[-1]

        duration = (dd_end - dd_start).days

        return {
            'max_drawdown_pct': max_dd,
            'max_drawdown_date': max_dd_date,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'duration_days': duration
        }

    def calmar_ratio(self) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
        cagr = self.cagr()
        max_dd = abs(self.max_drawdown()['max_drawdown_pct'])

        if max_dd == 0:
            return 0.0

        return cagr / max_dd

    def win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.0

        winning_trades = sum(1 for t in self.trades if getattr(t, 'pnl', 0) > 0)
        return winning_trades / len(self.trades) * 100

    def profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trades:
            return 0.0

        gross_profit = sum(getattr(t, 'pnl', 0) for t in self.trades if getattr(t, 'pnl', 0) > 0)
        gross_loss = abs(sum(getattr(t, 'pnl', 0) for t in self.trades if getattr(t, 'pnl', 0) < 0))

        if gross_loss == 0:
            return 0.0

        return gross_profit / gross_loss

    def expectancy(self) -> float:
        """Calculate average expected profit per trade"""
        if not self.trades:
            return 0.0

        total_pnl = sum(getattr(t, 'pnl', 0) for t in self.trades)
        return total_pnl / len(self.trades)

    def consecutive_wins_losses(self) -> Dict:
        """Calculate max consecutive wins and losses"""
        if not self.trades:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}

        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0

        for trade in self.trades:
            pnl = getattr(trade, 'pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }

    def recovery_factor(self) -> float:
        """Calculate recovery factor (Net Profit / Max Drawdown)"""
        net_profit = self.equity_curve.iloc[-1] - self.equity_curve.iloc[0]
        max_dd = abs(self.max_drawdown()['max_drawdown_pct']) / 100 * self.equity_curve.iloc[0]

        if max_dd == 0:
            return 0.0

        return net_profit / max_dd

    def ulcer_index(self) -> float:
        """Calculate Ulcer Index (Peter Martin)"""
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100
        squared_drawdown = drawdown ** 2
        ulcer = np.sqrt(squared_drawdown.mean())
        return ulcer

    def get_all_metrics(self) -> Dict:
        """Get all portfolio metrics"""
        max_dd_info = self.max_drawdown()
        consec_info = self.consecutive_wins_losses()

        metrics = {
            'total_return_pct': round(self.total_return(), 2),
            'cagr_pct': round(self.cagr(), 2),
            'volatility_pct': round(self.volatility(), 2),
            'sharpe_ratio': round(self.sharpe_ratio(), 2),
            'sortino_ratio': round(self.sortino_ratio(), 2),
            'max_drawdown_pct': round(max_dd_info['max_drawdown_pct'], 2),
            'max_drawdown_duration_days': max_dd_info['duration_days'],
            'calmar_ratio': round(self.calmar_ratio(), 2),
            'win_rate_pct': round(self.win_rate(), 2),
            'profit_factor': round(self.profit_factor(), 2),
            'expectancy': round(self.expectancy(), 2),
            'max_consecutive_wins': consec_info['max_consecutive_wins'],
            'max_consecutive_losses': consec_info['max_consecutive_losses'],
            'recovery_factor': round(self.recovery_factor(), 2),
            'ulcer_index': round(self.ulcer_index(), 2),
            'total_trades': len(self.trades)
        }

        return metrics

    def print_report(self):
        """Print formatted analytics report"""
        metrics = self.get_all_metrics()

        print("\n" + "=" * 70)
        print("PORTFOLIO PERFORMANCE REPORT")
        print("=" * 70)

        print("\n📊 RETURNS")
        print(f"  Total Return:           {metrics['total_return_pct']:>10.2f}%")
        print(f"  CAGR:                   {metrics['cagr_pct']:>10.2f}%")
        print(f"  Volatility (Annual):    {metrics['volatility_pct']:>10.2f}%")

        print("\n📈 RISK-ADJUSTED RETURNS")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:          {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:           {metrics['calmar_ratio']:>10.2f}")

        print("\n⚠️  RISK METRICS")
        print(f"  Max Drawdown:           {metrics['max_drawdown_pct']:>10.2f}%")
        print(f"  Drawdown Duration:      {metrics['max_drawdown_duration_days']:>10} days")
        print(f"  Ulcer Index:            {metrics['ulcer_index']:>10.2f}")
        print(f"  Recovery Factor:        {metrics['recovery_factor']:>10.2f}")

        print("\n💼 TRADING STATISTICS")
        print(f"  Total Trades:           {metrics['total_trades']:>10}")
        print(f"  Win Rate:               {metrics['win_rate_pct']:>10.2f}%")
        print(f"  Profit Factor:          {metrics['profit_factor']:>10.2f}")
        print(f"  Expectancy per Trade:   ${metrics['expectancy']:>10.2f}")
        print(f"  Max Consecutive Wins:   {metrics['max_consecutive_wins']:>10}")
        print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']:>10}")

        print("\n" + "=" * 70)


def create_equity_chart_ascii(equity_curve: pd.Series, height: int = 20, width: int = 60):
    """
    Create ASCII chart of equity curve

    Args:
        equity_curve: Equity time series
        height: Chart height in lines
        width: Chart width in characters
    """
    values = equity_curve.values
    min_val = values.min()
    max_val = values.max()

    # Normalize values to chart height
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1

    normalized = ((values - min_val) / range_val * (height - 1)).astype(int)

    # Downsample to fit width
    step = max(1, len(values) // width)
    sampled_indices = range(0, len(values), step)[:width]
    sampled_normalized = normalized[sampled_indices]

    # Create chart
    print("\nEquity Curve")
    print("=" * (width + 10))

    for row in range(height - 1, -1, -1):
        line = f"{max_val - (row * range_val / (height - 1)):>10.2f} │"
        for val in sampled_normalized:
            if val == row:
                line += "●"
            elif val > row:
                line += "│"
            else:
                line += " "
        print(line)

    print(" " * 11 + "└" + "─" * width)
    print(" " * 12 + "Start" + " " * (width - 20) + "End")


# Example usage
if __name__ == "__main__":
    # Generate sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = np.random.normal(0.001, 0.02, 252)
    equity = pd.Series((1 + returns).cumprod() * 100000, index=dates)

    # Create analytics
    analytics = PortfolioAnalytics(equity)

    # Print report
    analytics.print_report()

    # Create ASCII chart
    create_equity_chart_ascii(equity)
