"""
Risk Manager Agent
Applies risk controls and can veto or modify trading decisions
"""

from typing import Dict, Any
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent

    Responsibilities:
    - Enforce position size limits
    - Check exposure and concentration
    - Monitor drawdown
    - Apply stop-loss rules
    - Can veto or modify decisions
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_position_size: float = 10.0,
        max_exposure_pct: float = 0.5,
        max_drawdown_pct: float = 0.15,
        stop_loss_pct: float = 0.05
    ):
        super().__init__(
            name="Risk Manager",
            role="Risk Control and Position Sizing",
            llm=llm,
            temperature=0.2,  # Very low temperature for conservative risk management
            max_tokens=700
        )

        # Risk parameters
        self.max_position_size = max_position_size
        self.max_exposure_pct = max_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct

    def get_system_prompt(self) -> str:
        return f"""You are the Risk Manager for a trading firm.

Your responsibilities:
1. Enforce position size limits (Max: {self.max_position_size} units)
2. Monitor portfolio exposure (Max: {self.max_exposure_pct*100}% of capital)
3. Check drawdown levels (Max: {self.max_drawdown_pct*100}%)
4. Apply stop-loss rules ({self.stop_loss_pct*100}% stop)
5. Veto or modify risky decisions

You must respond in JSON format with the following structure:
{{
    "approved": true|false,
    "modified_size": 0-{self.max_position_size},
    "risk_assessment": "low|moderate|high|extreme",
    "violations": ["violation1", "violation2", ...],
    "warnings": ["warning1", "warning2", ...],
    "stop_loss_price": price or null,
    "take_profit_price": price or null,
    "recommendation": "approve|modify|reject",
    "reasoning": "Clear explanation of risk assessment"
}}

Be conservative. Protect capital. Better to miss opportunities than take excessive risk."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from decision and portfolio state"""

        # Extract information
        decision = context.get('decision', {})
        portfolio = context.get('portfolio', {})
        current_price = context.get('current_price', 0)
        current_position = context.get('current_position', {})

        # Calculate key risk metrics
        equity = portfolio.get('equity', 100000)
        cash = portfolio.get('cash', 100000)
        initial_capital = portfolio.get('initial_capital', 100000)

        current_drawdown = (equity - initial_capital) / initial_capital * 100
        position_value = abs(current_position.get('size', 0)) * current_price
        exposure_pct = position_value / equity * 100 if equity > 0 else 0

        # Build prompt
        prompt = f"""Evaluate the risk of the following trading decision:

PROPOSED DECISION:
- Action: {decision.get('decision', 'UNKNOWN')}
- Size: {decision.get('suggested_size', 0)} units
- Conviction: {decision.get('conviction', 'unknown')}
- Action Type: {decision.get('action_type', 'UNKNOWN')}

CURRENT PORTFOLIO STATE:
- Current Price: ${current_price:.2f}
- Total Equity: ${equity:.2f}
- Available Cash: ${cash:.2f}
- Current Position: {current_position.get('size', 0)} units
- Position Value: ${position_value:.2f}
- Current Exposure: {exposure_pct:.1f}%
- Unrealized P&L: ${current_position.get('unrealized_pnl', 0):.2f} ({current_position.get('unrealized_pnl_pct', 0):.1f}%)
- Current Drawdown: {current_drawdown:.1f}%

RISK LIMITS:
- Max Position Size: {self.max_position_size} units
- Max Exposure: {self.max_exposure_pct*100}% of equity
- Max Drawdown: {self.max_drawdown_pct*100}%
- Stop Loss: {self.stop_loss_pct*100}% from entry

RISK CHECKS:
1. Does proposed size exceed max position limit?
2. Would this trade exceed exposure limits?
3. Is portfolio approaching max drawdown?
4. Does current position need a stop-loss?
5. Is the risk/reward ratio acceptable?

Provide your risk assessment in the required JSON format.

If you reject or modify the decision, explain clearly why."""

        return prompt

    def _apply_hard_limits(
        self,
        decision: Dict[str, Any],
        portfolio: Dict[str, Any],
        current_position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply hard-coded risk limits (failsafe)

        Args:
            decision: CIO decision
            portfolio: Portfolio state
            current_position: Current position

        Returns:
            Dictionary with hard limit checks
        """
        violations = []
        warnings = []
        approved = True
        modified_size = decision.get('suggested_size', 0)

        # Check 1: Maximum position size
        if abs(modified_size) > self.max_position_size:
            violations.append(f"Position size {modified_size} exceeds max {self.max_position_size}")
            modified_size = self.max_position_size if modified_size > 0 else -self.max_position_size
            approved = False

        # Check 2: Drawdown limit
        equity = portfolio.get('equity', 100000)
        initial_capital = portfolio.get('initial_capital', 100000)
        current_drawdown_pct = (initial_capital - equity) / initial_capital

        if current_drawdown_pct > self.max_drawdown_pct:
            violations.append(f"Drawdown {current_drawdown_pct*100:.1f}% exceeds limit {self.max_drawdown_pct*100}%")
            modified_size = 0
            approved = False

        # Check 3: Stop loss on existing position
        if current_position.get('has_position'):
            unrealized_pnl_pct = abs(current_position.get('unrealized_pnl_pct', 0))
            if unrealized_pnl_pct > self.stop_loss_pct * 100:
                warnings.append(f"Position loss {unrealized_pnl_pct:.1f}% approaching stop-loss threshold")

        # Check 4: Available capital
        cash = portfolio.get('cash', 0)
        current_price = portfolio.get('current_price', 0)
        required_capital = abs(modified_size) * current_price

        if required_capital > cash and not current_position.get('has_position'):
            violations.append(f"Insufficient capital: need ${required_capital:.2f}, have ${cash:.2f}")
            # Reduce size to available capital
            if current_price > 0:
                modified_size = int(cash / current_price * 0.95)  # Use 95% to account for commission
            approved = False

        return {
            'approved': approved and len(violations) == 0,
            'modified_size': modified_size,
            'violations': violations,
            'warnings': warnings
        }

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process decision and apply risk controls

        Args:
            context: Dictionary containing:
                - decision: CIO decision
                - portfolio: Portfolio state
                - current_position: Current position
                - current_price: Current price

        Returns:
            Risk assessment with approved/modified decision
        """
        # First apply hard limits (failsafe)
        decision = context.get('decision', {})
        portfolio = context.get('portfolio', {})
        current_position = context.get('current_position', {})

        hard_limits = self._apply_hard_limits(decision, portfolio, current_position)

        # If hard limits already reject, return immediately
        if not hard_limits['approved'] and len(hard_limits['violations']) > 0:
            return {
                'agent_name': self.name,
                'agent_role': self.role,
                'approved': False,
                'modified_size': hard_limits['modified_size'],
                'risk_assessment': 'extreme',
                'violations': hard_limits['violations'],
                'warnings': hard_limits['warnings'],
                'stop_loss_price': None,
                'take_profit_price': None,
                'recommendation': 'reject',
                'reasoning': f"Hard limit violations: {', '.join(hard_limits['violations'])}"
            }

        # Get LLM risk assessment
        response = self.generate_response(context, use_json=True)

        # Merge hard limits with LLM response
        response['violations'] = list(set(response.get('violations', []) + hard_limits['violations']))
        response['warnings'] = list(set(response.get('warnings', []) + hard_limits['warnings']))

        # Override with hard limits if stricter
        if hard_limits['modified_size'] < response.get('modified_size', 0):
            response['modified_size'] = hard_limits['modified_size']

        # Validate response structure
        required_fields = ['approved', 'modified_size', 'risk_assessment', 'recommendation', 'reasoning']
        for field in required_fields:
            if field not in response:
                if field == 'approved':
                    response[field] = False
                elif field == 'modified_size':
                    response[field] = 0
                elif field == 'risk_assessment':
                    response[field] = 'high'
                elif field == 'recommendation':
                    response[field] = 'reject'
                elif field == 'reasoning':
                    response[field] = 'No reasoning provided'

        return response
