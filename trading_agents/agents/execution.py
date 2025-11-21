"""
Execution Agent
Converts approved decisions into concrete trade orders
"""

from typing import Dict, Any, Optional
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class ExecutionAgent(BaseAgent):
    """
    Execution Agent

    Responsibilities:
    - Convert decision into executable order
    - Determine optimal execution strategy
    - Handle order details (size, price, type)
    - Provide execution summary
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="Execution Trader",
            role="Order Execution and Trade Implementation",
            llm=llm,
            temperature=0.1,  # Very low temperature for precise execution
            max_tokens=500
        )

    def get_system_prompt(self) -> str:
        return """You are an Execution Trader for a trading firm.

Your responsibilities:
1. Convert approved decisions into concrete orders
2. Determine execution strategy (market, limit, etc.)
3. Calculate exact order size
4. Provide execution summary

You must respond in JSON format with the following structure:
{
    "action": "BUY|SELL|HOLD|CLOSE",
    "size": number (exact size to trade),
    "order_type": "MARKET|LIMIT",
    "price": number (execution price),
    "execution_strategy": "immediate|patient|scaled",
    "notes": "Brief execution notes",
    "summary": "Clear 1-2 sentence execution summary"
}

Be precise. Focus on execution mechanics, not strategy."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from approved decision"""

        # Extract information
        decision = context.get('decision', {})
        risk_approval = context.get('risk_approval', {})
        current_price = context.get('current_price', 0)
        current_position = context.get('current_position', {})

        # Determine actual action
        action_type = decision.get('action_type', 'HOLD')
        approved_size = risk_approval.get('modified_size', decision.get('suggested_size', 0))

        # Build prompt
        prompt = f"""Execute the following approved trading decision:

APPROVED DECISION:
- Decision: {decision.get('decision', 'UNKNOWN')}
- Action Type: {action_type}
- Approved Size: {approved_size} units
- Risk Assessment: {risk_approval.get('risk_assessment', 'unknown')}

CURRENT STATE:
- Current Price: ${current_price:.2f}
- Current Position: {current_position.get('size', 0)} units
- Has Position: {current_position.get('has_position', False)}

RISK PARAMETERS:
- Stop Loss: {risk_approval.get('stop_loss_price', 'None')}
- Take Profit: {risk_approval.get('take_profit_price', 'None')}

Convert this into a concrete order and provide execution details in the required JSON format.

Guidelines:
- If approved_size is 0 or decision is HOLD, action should be HOLD
- If we have a position and decision is opposite direction, action should be CLOSE
- For opening new positions, action should be BUY or SELL
- Use MARKET order type for simplicity (can be enhanced later)
- Price should be current market price"""

        return prompt

    def _create_simple_order(
        self,
        decision: Dict[str, Any],
        risk_approval: Dict[str, Any],
        current_price: float,
        current_position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create simple order without LLM (fallback or fast execution)

        Args:
            decision: CIO decision
            risk_approval: Risk manager approval
            current_price: Current market price
            current_position: Current position state

        Returns:
            Order dictionary
        """
        approved_size = risk_approval.get('modified_size', decision.get('suggested_size', 0))
        decision_type = decision.get('decision', 'HOLD')
        has_position = current_position.get('has_position', False)

        # Determine action
        if not risk_approval.get('approved', False) or approved_size == 0:
            action = 'HOLD'
            size = 0
        elif decision_type == 'HOLD':
            action = 'HOLD'
            size = 0
        elif has_position:
            # Close existing position
            action = 'CLOSE'
            size = 0  # Will be handled by market env
        elif decision_type == 'BUY':
            action = 'BUY'
            size = approved_size
        elif decision_type == 'SELL':
            action = 'SELL'
            size = -approved_size
        else:
            action = 'HOLD'
            size = 0

        return {
            'action': action,
            'size': size,
            'order_type': 'MARKET',
            'price': current_price,
            'execution_strategy': 'immediate',
            'notes': f'Executing {action} order for {abs(size)} units',
            'summary': f'Execute {action} at market price ${current_price:.2f}'
        }

    def process(self, context: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
        """
        Process approved decision and create execution order

        Args:
            context: Dictionary containing:
                - decision: CIO decision
                - risk_approval: Risk manager approval
                - current_price: Current price
                - current_position: Current position
            use_llm: Whether to use LLM for execution (default True)

        Returns:
            Execution order dictionary
        """
        # Check if trade was approved
        risk_approval = context.get('risk_approval', {})
        if not risk_approval.get('approved', False):
            return {
                'agent_name': self.name,
                'agent_role': self.role,
                'action': 'HOLD',
                'size': 0,
                'order_type': 'NONE',
                'price': context.get('current_price', 0),
                'execution_strategy': 'none',
                'notes': 'Trade rejected by risk manager',
                'summary': 'No execution: trade was rejected by risk management'
            }

        # Use simple order creation for speed and reliability
        if not use_llm:
            order = self._create_simple_order(
                context.get('decision', {}),
                risk_approval,
                context.get('current_price', 0),
                context.get('current_position', {})
            )
            order['agent_name'] = self.name
            order['agent_role'] = self.role
            return order

        # Use LLM for execution planning
        response = self.generate_response(context, use_json=True)

        # Validate response structure
        required_fields = ['action', 'size', 'order_type', 'price', 'summary']
        for field in required_fields:
            if field not in response:
                if field == 'action':
                    response[field] = 'HOLD'
                elif field == 'size':
                    response[field] = 0
                elif field == 'order_type':
                    response[field] = 'MARKET'
                elif field == 'price':
                    response[field] = context.get('current_price', 0)
                elif field == 'summary':
                    response[field] = 'No execution summary'

        return response
