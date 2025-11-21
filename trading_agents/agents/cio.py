"""
CIO (Chief Investment Officer) / Decision Agent
Makes final trading decisions based on all inputs
"""

from typing import Dict, Any
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class CIOAgent(BaseAgent):
    """
    CIO / Decision Agent

    Responsibilities:
    - Consolidate all analyst inputs
    - Make final BUY/SELL/HOLD decision
    - Provide clear justification
    - Consider risk/reward balance
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="Chief Investment Officer",
            role="Final Trading Decision",
            llm=llm,
            temperature=0.4,
            max_tokens=800
        )

    def get_system_prompt(self) -> str:
        return """You are the Chief Investment Officer (CIO) of a trading firm.

Your responsibilities:
1. Review all analyst recommendations
2. Make final trading decision: BUY, SELL, or HOLD
3. Determine conviction level
4. Provide clear, concise justification
5. Consider risk/reward balance

You must respond in JSON format with the following structure:
{
    "decision": "BUY|SELL|HOLD",
    "conviction": "high|medium|low",
    "suggested_size": 0-10,
    "action_type": "OPEN|CLOSE|HOLD|ADD|REDUCE",
    "primary_reason": "Main reason for decision",
    "supporting_factors": ["factor1", "factor2", ...],
    "risk_considerations": ["risk1", "risk2", ...],
    "confidence": 0.0-1.0,
    "justification": "Clear 3-4 sentence explanation of final decision"
}

Be decisive but prudent. Consider all inputs but make your own judgment. Clearly articulate your reasoning."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from all analyst inputs"""

        # Extract information
        market_analysis = context.get('market_analysis', {})
        news_analysis = context.get('news_analysis', {})
        strategy = context.get('strategy', {})
        current_position = context.get('current_position', {})
        current_price = context.get('current_price', 0)

        # Build comprehensive prompt
        prompt = f"""Make a final trading decision based on all analyst inputs:

CURRENT MARKET SITUATION:
- Price: ${current_price:.2f}
- Current Position: {current_position.get('size', 0)} units
- Unrealized P&L: ${current_position.get('unrealized_pnl', 0):.2f} ({current_position.get('unrealized_pnl_pct', 0):.2f}%)

MARKET ANALYST ASSESSMENT:
- Trend: {market_analysis.get('trend', 'unknown')} ({market_analysis.get('trend_strength', 'unknown')})
- Momentum: {market_analysis.get('momentum', 'unknown')}
- Volatility: {market_analysis.get('volatility', 'unknown')}
- Confidence: {market_analysis.get('confidence', 0):.2f}
- Summary: {market_analysis.get('summary', 'N/A')}

NEWS & EVENTS ANALYST ASSESSMENT:
- Sentiment: {news_analysis.get('sentiment', 'neutral')} (Score: {news_analysis.get('sentiment_score', 0):.2f})
- Catalyst Score: {news_analysis.get('catalyst_score', 0):.2f}
- Impact: {news_analysis.get('impact_assessment', 'none')}
- Recommendation: {news_analysis.get('recommendation', 'proceed')}
- Summary: {news_analysis.get('summary', 'N/A')}

STRATEGY ANALYST RECOMMENDATION:
- Direction: {strategy.get('direction', 'NEUTRAL')}
- Long Probability: {strategy.get('probability_long', 0):.2f}
- Short Probability: {strategy.get('probability_short', 0):.2f}
- Suggested Size: {strategy.get('suggested_size', 0)}
- Strategy Type: {strategy.get('strategy_type', 'unknown')}
- Confidence: {strategy.get('confidence', 0):.2f}
- Reasoning: {strategy.get('reasoning', 'N/A')}

As CIO, make your final decision and provide your justification in the required JSON format.

Remember:
- HOLD means take no action
- BUY means open or add to long position
- SELL means open short or close long position
- Consider the full picture, not just individual recommendations
- Be honest about conviction level"""

        return prompt

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all inputs and make final decision

        Args:
            context: Dictionary containing:
                - market_analysis: Market Analyst output
                - news_analysis: News Analyst output
                - strategy: Strategy Analyst output
                - current_position: Current position info
                - current_price: Current price

        Returns:
            Decision dictionary with BUY/SELL/HOLD and justification
        """
        response = self.generate_response(context, use_json=True)

        # Validate response structure
        required_fields = [
            'decision', 'conviction', 'suggested_size',
            'action_type', 'confidence', 'justification'
        ]
        for field in required_fields:
            if field not in response:
                if field == 'decision':
                    response[field] = 'HOLD'
                elif field == 'conviction':
                    response[field] = 'low'
                elif field == 'suggested_size':
                    response[field] = 0
                elif field == 'action_type':
                    response[field] = 'HOLD'
                elif field == 'confidence':
                    response[field] = 0.5
                elif field == 'justification':
                    response[field] = 'No justification provided'

        return response
