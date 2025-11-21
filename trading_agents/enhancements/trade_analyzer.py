"""
Trade Performance Analyzer
Detailed analysis of individual trades and trading patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta


class TradeAnalyzer:
    """Analyze trading performance and patterns"""

    def __init__(self, trades: List):
        """
        Initialize trade analyzer

        Args:
            trades: List of Trade objects
        """
        self.trades = trades

    def get_trade_stats(self) -> Dict:
        """Get comprehensive trade statistics"""
        if not self.trades:
            return {}

        pnls = [t.pnl for t in self.trades]
        sizes = [abs(t.size) for t in self.trades]

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'largest_win': max(pnls),
            'largest_loss': min(pnls),
            'avg_trade_size': np.mean(sizes),
            'total_pnl': sum(pnls),
            'profit_factor': sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0
        }

    def get_holding_period_stats(self) -> Dict:
        """Analyze holding periods"""
        if len(self.trades) < 2:
            return {}

        # Group trades by pairs (entry/exit)
        holding_periods = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                duration = (self.trades[i + 1].timestamp - self.trades[i].timestamp).total_seconds() / 3600
                holding_periods.append(duration)

        if not holding_periods:
            return {}

        return {
            'avg_holding_hours': np.mean(holding_periods),
            'min_holding_hours': min(holding_periods),
            'max_holding_hours': max(holding_periods),
            'median_holding_hours': np.median(holding_periods)
        }

    def get_time_analysis(self) -> Dict:
        """Analyze trading by time of day/week"""
        if not self.trades:
            return {}

        df = pd.DataFrame([{
            'hour': t.timestamp.hour,
            'day': t.timestamp.weekday(),
            'pnl': t.pnl
        } for t in self.trades])

        hourly_pnl = df.groupby('hour')['pnl'].sum().to_dict()
        daily_pnl = df.groupby('day')['pnl'].sum().to_dict()

        return {
            'best_hour': max(hourly_pnl.items(), key=lambda x: x[1])[0] if hourly_pnl else None,
            'worst_hour': min(hourly_pnl.items(), key=lambda x: x[1])[0] if hourly_pnl else None,
            'best_day': max(daily_pnl.items(), key=lambda x: x[1])[0] if daily_pnl else None,
            'worst_day': min(daily_pnl.items(), key=lambda x: x[1])[0] if daily_pnl else None
        }

    def print_report(self):
        """Print detailed trade analysis report"""
        stats = self.get_trade_stats()
        holding = self.get_holding_period_stats()
        time_analysis = self.get_time_analysis()

        print("\n" + "=" * 70)
        print("TRADE PERFORMANCE ANALYSIS")
        print("=" * 70)

        print("\n📊 OVERALL STATISTICS")
        print(f"  Total Trades:           {stats.get('total_trades', 0):>10}")
        print(f"  Winning Trades:         {stats.get('winning_trades', 0):>10}")
        print(f"  Losing Trades:          {stats.get('losing_trades', 0):>10}")
        print(f"  Win Rate:               {stats.get('win_rate', 0):>10.1f}%")

        print("\n💰 P&L METRICS")
        print(f"  Total P&L:              ${stats.get('total_pnl', 0):>10.2f}")
        print(f"  Average Win:            ${stats.get('avg_win', 0):>10.2f}")
        print(f"  Average Loss:           ${stats.get('avg_loss', 0):>10.2f}")
        print(f"  Win/Loss Ratio:         {stats.get('win_loss_ratio', 0):>10.2f}")
        print(f"  Largest Win:            ${stats.get('largest_win', 0):>10.2f}")
        print(f"  Largest Loss:           ${stats.get('largest_loss', 0):>10.2f}")
        print(f"  Profit Factor:          {stats.get('profit_factor', 0):>10.2f}")

        if holding:
            print("\n⏱️  HOLDING PERIODS")
            print(f"  Average:                {holding.get('avg_holding_hours', 0):>10.1f} hours")
            print(f"  Minimum:                {holding.get('min_holding_hours', 0):>10.1f} hours")
            print(f"  Maximum:                {holding.get('max_holding_hours', 0):>10.1f} hours")

        if time_analysis:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            best_day = time_analysis.get('best_day')
            worst_day = time_analysis.get('worst_day')

            print("\n📅 TIME ANALYSIS")
            if best_day is not None:
                print(f"  Best Day:               {day_names[best_day]:>10}")
            if worst_day is not None:
                print(f"  Worst Day:              {day_names[worst_day]:>10}")
            if time_analysis.get('best_hour') is not None:
                print(f"  Best Hour:              {time_analysis.get('best_hour'):>10}:00")

        print("\n" + "=" * 70)


# Quick functions
def analyze_trades(trades: List) -> Dict:
    """Quick trade analysis"""
    analyzer = TradeAnalyzer(trades)
    return analyzer.get_trade_stats()
