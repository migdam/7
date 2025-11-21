"""
Research/Strategy Agent
Generates trade ideas based on pattern recognition and analysis
"""

from typing import Dict, Any
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class StrategyAgent(BaseAgent):
    """
    Research/Strategy Agent

    Responsibilities:
    - Synthesize market analysis and news
    - Identify trading patterns
    - Generate initial directional view
    - Suggest position size
    - Provide reasoning
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="Research/Strategy Analyst",
            role="Pattern Recognition and Strategy Development",
            llm=llm,
            temperature=0.5,
            max_tokens=1000
        )

    def get_system_prompt(self) -> str:
        return """You are a professional Research and Strategy Analyst for a trading firm.

Your responsibilities:
1. Synthesize technical analysis and news/events
2. Identify trading patterns and opportunities
3. Generate directional view with probability
4. Suggest initial position size
5. Provide clear reasoning

You must respond in JSON format with the following structure:
{
    "direction": "LONG|SHORT|NEUTRAL",
    "probability_long": 0.0-1.0,
    "probability_short": 0.0-1.0,
    "suggested_size": 0-10,
    "strategy_type": "trend_following|mean_reversion|breakout|other",
    "entry_logic": "Brief explanation of why to enter",
    "key_patterns": ["pattern1", "pattern2", ...],
    "confluence_factors": ["factor1", "factor2", ...],
    "concerns": ["concern1", "concern2", ...],
    "confidence": 0.0-1.0,
    "reasoning": "Detailed 3-4 sentence explanation"
}

Be analytical. Consider both technical and fundamental factors. Be honest about uncertainty."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from market analysis and news"""

        # Extract information
        market_analysis = context.get('market_analysis', {})
        news_analysis = context.get('news_analysis', {})
        current_position = context.get('current_position', {})
        current_price = context.get('current_price', 0)

        # Build prompt
        prompt = f"""Develop a trading strategy based on the following analysis:

CURRENT PRICE: ${current_price:.2f}

MARKET ANALYSIS (from Market Analyst):
- Trend: {market_analysis.get('trend', 'unknown')} ({market_analysis.get('trend_strength', 'unknown')})
- Momentum: {market_analysis.get('momentum', 'unknown')}
- Volatility: {market_analysis.get('volatility', 'unknown')}
- Confidence: {market_analysis.get('confidence', 0)}
- Summary: {market_analysis.get('summary', 'No summary available')}
- Key Observations: {', '.join(market_analysis.get('key_observations', []))}

NEWS & EVENTS ANALYSIS (from News Analyst):
- Sentiment: {news_analysis.get('sentiment', 'neutral')} (Score: {news_analysis.get('sentiment_score', 0)})
- Catalyst Score: {news_analysis.get('catalyst_score', 0)}
- Impact: {news_analysis.get('impact_assessment', 'none')}
- Recommendation: {news_analysis.get('recommendation', 'proceed')}
- Risk Events: {', '.join(news_analysis.get('risk_events', []))}
- Summary: {news_analysis.get('summary', 'No summary available')}

CURRENT POSITION:
- Has Position: {current_position.get('has_position', False)}
- Size: {current_position.get('size', 0)}
- Unrealized P&L: {current_position.get('unrealized_pnl', 0):.2f}

Based on this information, develop a trading strategy and provide your recommendation in the required JSON format."""

        return prompt

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analyses and generate strategy

        Args:
            context: Dictionary containing:
                - market_analysis: Output from Market Analyst
                - news_analysis: Output from News Analyst
                - current_position: Current position info
                - current_price: Current price

        Returns:
            Strategy dictionary with direction, size, reasoning
        """
        response = self.generate_response(context, use_json=True)

        # Validate response structure
        required_fields = [
            'direction', 'probability_long', 'probability_short',
            'suggested_size', 'confidence', 'reasoning'
        ]
        for field in required_fields:
            if field not in response:
                if 'probability' in field or field == 'confidence':
                    response[field] = 0.5
                elif field == 'suggested_size':
                    response[field] = 0
                elif field == 'direction':
                    response[field] = 'NEUTRAL'
                elif field == 'reasoning':
                    response[field] = 'No reasoning provided'

        return response
