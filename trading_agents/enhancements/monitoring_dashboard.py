"""
Real-time Monitoring CLI Dashboard
Live display of trading activity, positions, and metrics
"""

import time
from datetime import datetime
from typing import Dict, Optional
import os


class TradingDashboard:
    """CLI-based real-time trading dashboard"""

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize dashboard

        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.last_update = None

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def render_header(self):
        """Render dashboard header"""
        print("=" * 80)
        print(" " * 20 + "TRADING AGENTS - LIVE DASHBOARD")
        print("=" * 80)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def render_portfolio(self, portfolio_state: Dict):
        """
        Render portfolio section

        Args:
            portfolio_state: Portfolio state dictionary
        """
        equity = portfolio_state.get('equity', 0)
        cash = portfolio_state.get('cash', 0)
        position = portfolio_state.get('position', {})

        print("┌─ PORTFOLIO " + "─" * 66 + "┐")
        print(f"│ Equity:        ${equity:>15,.2f}                                      │")
        print(f"│ Cash:          ${cash:>15,.2f}                                      │")
        print(f"│ Position:      {position.get('size', 0):>7.2f} units                                   │")
        print(f"│ Market Value:  ${position.get('market_value', 0):>15,.2f}                                      │")
        print(f"│ Unrealized P&L: ${position.get('unrealized_pnl', 0):>15,.2f} ({position.get('unrealized_pnl_pct', 0):>6.2f}%)              │")
        print("└" + "─" * 78 + "┘")
        print()

    def render_latest_decision(self, decision_data: Dict):
        """
        Render latest agent decision

        Args:
            decision_data: Latest decision data
        """
        print("┌─ LATEST DECISION " + "─" * 60 + "┐")

        market_analysis = decision_data.get('market_analysis', {})
        cio_decision = decision_data.get('cio_decision', {})
        risk_approval = decision_data.get('risk_approval', {})

        print(f"│ Market Trend:  {market_analysis.get('trend', 'N/A'):<20} Momentum: {market_analysis.get('momentum', 'N/A'):<15} │")
        print(f"│ CIO Decision:  {cio_decision.get('decision', 'N/A'):<20} Conviction: {cio_decision.get('conviction', 'N/A'):<15} │")
        print(f"│ Risk Status:   {'APPROVED' if risk_approval.get('approved') else 'REJECTED':<20}                       │")
        print(f"│ Justification: {(cio_decision.get('justification', 'N/A')[:60]):<60} │")
        print("└" + "─" * 78 + "┘")
        print()

    def render_metrics(self, metrics: Dict):
        """
        Render performance metrics

        Args:
            metrics: Performance metrics dictionary
        """
        print("┌─ PERFORMANCE METRICS " + "─" * 54 + "┐")
        print(f"│ Total Return:    {metrics.get('total_return_pct', 0):>6.2f}%  │  Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):>6.2f}     │")
        print(f"│ Max Drawdown:    {metrics.get('max_drawdown_pct', 0):>6.2f}%  │  Win Rate:      {metrics.get('win_rate', 0)*100:>6.1f}%    │")
        print(f"│ Total Trades:    {metrics.get('total_trades', 0):>6}      │  Profit Factor: {metrics.get('profit_factor', 0):>6.2f}     │")
        print("└" + "─" * 78 + "┘")
        print()

    def render_recent_trades(self, trades: list, limit: int = 5):
        """
        Render recent trades

        Args:
            trades: List of recent trades
            limit: Number of trades to show
        """
        print("┌─ RECENT TRADES " + "─" * 61 + "┐")
        print("│ Time             │ Side │ Size   │ Price    │ P&L      │ P&L %   │")
        print("├" + "─" * 78 + "┤")

        recent = trades[-limit:] if len(trades) > limit else trades

        for trade in reversed(recent):
            time_str = trade.timestamp.strftime('%H:%M:%S') if hasattr(trade.timestamp, 'strftime') else str(trade.timestamp)
            side = trade.side[:4]
            pnl_color = '+' if trade.pnl >= 0 else ''

            print(f"│ {time_str:<16} │ {side:<4} │ {trade.size:>6.2f} │ ${trade.price:>7.2f} │ {pnl_color}${trade.pnl:>7.2f} │ {pnl_color}{trade.pnl_pct:>6.2f}% │")

        print("└" + "─" * 78 + "┘")
        print()

    def render_alerts(self, alerts: list):
        """
        Render alerts/warnings

        Args:
            alerts: List of alert messages
        """
        if not alerts:
            return

        print("┌─ ALERTS " + "─" * 68 + "┐")
        for alert in alerts[-5:]:  # Show last 5 alerts
            print(f"│ ⚠  {alert[:74]:<74} │")
        print("└" + "─" * 78 + "┘")
        print()

    def render_full_dashboard(
        self,
        portfolio_state: Dict,
        decision_data: Dict,
        metrics: Dict,
        trades: list,
        alerts: Optional[list] = None
    ):
        """
        Render complete dashboard

        Args:
            portfolio_state: Portfolio state
            decision_data: Latest decision
            metrics: Performance metrics
            trades: Trade history
            alerts: Alert messages
        """
        self.clear_screen()
        self.render_header()
        self.render_portfolio(portfolio_state)
        self.render_latest_decision(decision_data)
        self.render_metrics(metrics)
        self.render_recent_trades(trades)

        if alerts:
            self.render_alerts(alerts)

        print("Press Ctrl+C to stop")

    def monitor_live(self, data_source_func, duration: int = None):
        """
        Monitor live trading activity

        Args:
            data_source_func: Function that returns dashboard data
            duration: Duration in seconds (None for infinite)
        """
        start_time = time.time()

        try:
            while True:
                # Get latest data
                data = data_source_func()

                # Render dashboard
                self.render_full_dashboard(
                    portfolio_state=data.get('portfolio', {}),
                    decision_data=data.get('decision', {}),
                    metrics=data.get('metrics', {}),
                    trades=data.get('trades', []),
                    alerts=data.get('alerts', [])
                )

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Wait for next update
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user")


# Simple ASCII chart
def render_ascii_chart(values: list, height: int = 10, width: int = 60):
    """
    Render simple ASCII line chart

    Args:
        values: List of values to plot
        height: Chart height
        width: Chart width
    """
    if not values:
        return

    # Normalize values
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]

    # Downsample to fit width
    step = max(1, len(normalized) // width)
    sampled = normalized[::step][:width]

    # Render
    print("┌" + "─" * (width + 2) + "┐")
    for row in range(height - 1, -1, -1):
        line = "│ "
        for val in sampled:
            if val == row:
                line += "●"
            elif val > row:
                line += "│"
            else:
                line += " "
        line += " " * (width - len(sampled) + 1) + "│"
        print(line)
    print("└" + "─" * (width + 2) + "┘")


# Example usage
if __name__ == "__main__":
    # Mock data source
    def get_mock_data():
        from trading_agents.core.market_env import Trade
        from datetime import datetime

        return {
            'portfolio': {
                'equity': 105230.50,
                'cash': 95430.25,
                'position': {
                    'size': 10.5,
                    'market_value': 9800.25,
                    'unrealized_pnl': 523.50,
                    'unrealized_pnl_pct': 5.65
                }
            },
            'decision': {
                'market_analysis': {
                    'trend': 'bullish',
                    'momentum': 'positive'
                },
                'cio_decision': {
                    'decision': 'HOLD',
                    'conviction': 'medium',
                    'justification': 'Market conditions favorable but awaiting confirmation'
                },
                'risk_approval': {
                    'approved': True
                }
            },
            'metrics': {
                'total_return_pct': 5.23,
                'max_drawdown_pct': -2.15,
                'sharpe_ratio': 1.85,
                'win_rate': 0.62,
                'total_trades': 45,
                'profit_factor': 1.95
            },
            'trades': [
                Trade('ASSET', 'BUY', 5, 100.0, datetime.now(), pnl=50.0, pnl_pct=5.0),
                Trade('ASSET', 'SELL', 5, 105.0, datetime.now(), pnl=25.0, pnl_pct=2.5)
            ],
            'alerts': ['Position size approaching limit']
        }

    # Create dashboard
    dashboard = TradingDashboard(update_interval=2.0)

    # Render once
    data = get_mock_data()
    dashboard.render_full_dashboard(
        portfolio_state=data['portfolio'],
        decision_data=data['decision'],
        metrics=data['metrics'],
        trades=data['trades'],
        alerts=data['alerts']
    )
