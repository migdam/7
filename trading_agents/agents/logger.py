"""
Trade Journal / Logger Agent
Logs all decisions, rationales, and outcomes for transparency
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class TradeLoggerAgent:
    """
    Trade Journal / Logger Agent

    Responsibilities:
    - Log all agent decisions
    - Record trade rationales
    - Track P&L and equity
    - Export to JSON/CSV
    - Provide audit trail
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logger

        Args:
            log_dir: Directory for log files
        """
        self.name = "Trade Journal"
        self.role = "Decision Logging and Audit Trail"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Log storage
        self.decision_log: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.equity_log: List[Dict[str, Any]] = []

        # Create session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_decision(
        self,
        timestamp: datetime,
        price: float,
        market_analysis: Dict[str, Any],
        news_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
        cio_decision: Dict[str, Any],
        risk_approval: Dict[str, Any],
        execution_order: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ):
        """
        Log a complete decision cycle

        Args:
            timestamp: Decision timestamp
            price: Current price
            market_analysis: Market analyst output
            news_analysis: News analyst output
            strategy: Strategy analyst output
            cio_decision: CIO decision output
            risk_approval: Risk manager output
            execution_order: Execution order
            portfolio_state: Current portfolio state
        """
        entry = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'session_id': self.session_id,
            'price': price,
            'market_analysis': {
                'trend': market_analysis.get('trend'),
                'momentum': market_analysis.get('momentum'),
                'volatility': market_analysis.get('volatility'),
                'confidence': market_analysis.get('confidence'),
                'summary': market_analysis.get('summary')
            },
            'news_analysis': {
                'sentiment': news_analysis.get('sentiment'),
                'sentiment_score': news_analysis.get('sentiment_score'),
                'catalyst_score': news_analysis.get('catalyst_score'),
                'impact': news_analysis.get('impact_assessment'),
                'recommendation': news_analysis.get('recommendation')
            },
            'strategy': {
                'direction': strategy.get('direction'),
                'probability_long': strategy.get('probability_long'),
                'suggested_size': strategy.get('suggested_size'),
                'confidence': strategy.get('confidence'),
                'reasoning': strategy.get('reasoning')
            },
            'cio_decision': {
                'decision': cio_decision.get('decision'),
                'conviction': cio_decision.get('conviction'),
                'suggested_size': cio_decision.get('suggested_size'),
                'action_type': cio_decision.get('action_type'),
                'justification': cio_decision.get('justification')
            },
            'risk_approval': {
                'approved': risk_approval.get('approved'),
                'modified_size': risk_approval.get('modified_size'),
                'risk_assessment': risk_approval.get('risk_assessment'),
                'violations': risk_approval.get('violations', []),
                'reasoning': risk_approval.get('reasoning')
            },
            'execution': {
                'action': execution_order.get('action'),
                'size': execution_order.get('size'),
                'price': execution_order.get('price'),
                'summary': execution_order.get('summary')
            },
            'portfolio': {
                'equity': portfolio_state.get('equity'),
                'cash': portfolio_state.get('cash'),
                'position_size': portfolio_state.get('position', {}).get('size', 0),
                'unrealized_pnl': portfolio_state.get('position', {}).get('unrealized_pnl', 0)
            }
        }

        self.decision_log.append(entry)

    def log_trade(
        self,
        timestamp: datetime,
        action: str,
        size: float,
        price: float,
        pnl: float = 0.0,
        rationale: str = ""
    ):
        """
        Log a completed trade

        Args:
            timestamp: Trade timestamp
            action: Trade action (BUY/SELL/CLOSE)
            size: Trade size
            price: Execution price
            pnl: Realized P&L (if closing)
            rationale: Trade rationale
        """
        entry = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'session_id': self.session_id,
            'action': action,
            'size': size,
            'price': price,
            'pnl': pnl,
            'rationale': rationale
        }

        self.trade_log.append(entry)

    def log_equity(
        self,
        timestamp: datetime,
        equity: float,
        cash: float,
        position_value: float
    ):
        """
        Log equity curve point

        Args:
            timestamp: Timestamp
            equity: Total equity
            cash: Cash balance
            position_value: Position market value
        """
        entry = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'session_id': self.session_id,
            'equity': equity,
            'cash': cash,
            'position_value': position_value
        }

        self.equity_log.append(entry)

    def save_logs(self, prefix: str = ""):
        """
        Save all logs to files

        Args:
            prefix: Optional prefix for log files
        """
        timestamp = self.session_id if not prefix else f"{prefix}_{self.session_id}"

        # Save decision log as JSON
        decision_file = self.log_dir / f"{timestamp}_decisions.json"
        with open(decision_file, 'w') as f:
            json.dump(self.decision_log, f, indent=2)

        # Save trade log as CSV
        if self.trade_log:
            trade_file = self.log_dir / f"{timestamp}_trades.csv"
            with open(trade_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trade_log[0].keys())
                writer.writeheader()
                writer.writerows(self.trade_log)

        # Save equity log as CSV
        if self.equity_log:
            equity_file = self.log_dir / f"{timestamp}_equity.csv"
            with open(equity_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.equity_log[0].keys())
                writer.writeheader()
                writer.writerows(self.equity_log)

        print(f"Logs saved to {self.log_dir}/ with prefix '{timestamp}'")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics

        Returns:
            Dictionary with summary stats
        """
        total_decisions = len(self.decision_log)
        total_trades = len(self.trade_log)

        # Count decisions by type
        buy_decisions = sum(1 for d in self.decision_log if d['cio_decision']['decision'] == 'BUY')
        sell_decisions = sum(1 for d in self.decision_log if d['cio_decision']['decision'] == 'SELL')
        hold_decisions = sum(1 for d in self.decision_log if d['cio_decision']['decision'] == 'HOLD')

        # Count approved vs rejected
        approved = sum(1 for d in self.decision_log if d['risk_approval']['approved'])
        rejected = total_decisions - approved

        # Calculate P&L from trades
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_log)
        winning_trades = sum(1 for t in self.trade_log if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in self.trade_log if t.get('pnl', 0) < 0)

        # Equity stats
        if self.equity_log:
            initial_equity = self.equity_log[0]['equity']
            final_equity = self.equity_log[-1]['equity']
            total_return = (final_equity - initial_equity) / initial_equity * 100
        else:
            total_return = 0.0

        return {
            'session_id': self.session_id,
            'total_decisions': total_decisions,
            'total_trades': total_trades,
            'buy_decisions': buy_decisions,
            'sell_decisions': sell_decisions,
            'hold_decisions': hold_decisions,
            'approved_decisions': approved,
            'rejected_decisions': rejected,
            'total_pnl': total_pnl,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_return_pct': total_return
        }

    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {summary['session_id']}")
        print(f"\nDecisions:")
        print(f"  Total: {summary['total_decisions']}")
        print(f"  Buy: {summary['buy_decisions']}")
        print(f"  Sell: {summary['sell_decisions']}")
        print(f"  Hold: {summary['hold_decisions']}")
        print(f"  Approved: {summary['approved_decisions']}")
        print(f"  Rejected: {summary['rejected_decisions']}")
        print(f"\nTrades:")
        print(f"  Total: {summary['total_trades']}")
        print(f"  Winning: {summary['winning_trades']}")
        print(f"  Losing: {summary['losing_trades']}")
        print(f"  Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"\nPerformance:")
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print(f"  Total Return: {summary['total_return_pct']:.2f}%")
        print("="*60)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and log a complete decision cycle

        Args:
            context: Dictionary with all agent outputs

        Returns:
            Log confirmation
        """
        self.log_decision(
            timestamp=context.get('timestamp', datetime.now()),
            price=context.get('current_price', 0),
            market_analysis=context.get('market_analysis', {}),
            news_analysis=context.get('news_analysis', {}),
            strategy=context.get('strategy', {}),
            cio_decision=context.get('cio_decision', {}),
            risk_approval=context.get('risk_approval', {}),
            execution_order=context.get('execution_order', {}),
            portfolio_state=context.get('portfolio_state', {})
        )

        # Log equity
        portfolio = context.get('portfolio_state', {})
        self.log_equity(
            timestamp=context.get('timestamp', datetime.now()),
            equity=portfolio.get('equity', 0),
            cash=portfolio.get('cash', 0),
            position_value=portfolio.get('position', {}).get('market_value', 0)
        )

        return {
            'agent_name': self.name,
            'agent_role': self.role,
            'logged': True,
            'total_entries': len(self.decision_log)
        }

    def __repr__(self) -> str:
        return f"TradeLoggerAgent(session_id='{self.session_id}', decisions={len(self.decision_log)})"
