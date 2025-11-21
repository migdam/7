"""
Market Analyst Agent
Analyzes short-term trends, volatility, and technical indicators
"""

from typing import Dict, Any
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class MarketAnalystAgent(BaseAgent):
    """
    Market Analyst Agent

    Responsibilities:
    - Analyze price trends
    - Assess momentum and volatility
    - Evaluate technical indicators
    - Provide market condition summary
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="Market Analyst",
            role="Technical Analysis and Market Conditions",
            llm=llm,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=800
        )

    def get_system_prompt(self) -> str:
        return """You are a professional Market Analyst for a trading firm.

Your responsibilities:
1. Analyze price trends and momentum
2. Evaluate technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. Assess market volatility and strength
4. Provide clear, concise market condition summary

You must respond in JSON format with the following structure:
{
    "trend": "bullish|bearish|neutral",
    "trend_strength": "strong|moderate|weak",
    "momentum": "positive|negative|neutral",
    "volatility": "high|moderate|low",
    "key_observations": ["observation1", "observation2", ...],
    "support_levels": [price1, price2],
    "resistance_levels": [price1, price2],
    "confidence": 0.0-1.0,
    "summary": "Brief 2-3 sentence summary"
}

Be objective and data-driven. Focus on what the indicators show, not predictions."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from market data"""

        # Extract key information
        current_price = context.get('current_price', 0)
        indicators = context.get('indicators', {})
        price_history = context.get('price_history', [])

        # Build structured prompt
        prompt = f"""Analyze the current market conditions:

CURRENT PRICE: ${current_price:.2f}

TECHNICAL INDICATORS:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- MACD Signal: {indicators.get('macd_signal', 'N/A')}
- MACD Histogram: {indicators.get('macd_histogram', 'N/A')}
- SMA 20: {indicators.get('sma_20', 'N/A')}
- SMA 50: {indicators.get('sma_50', 'N/A')}
- Bollinger Upper: {indicators.get('bb_upper', 'N/A')}
- Bollinger Middle: {indicators.get('bb_middle', 'N/A')}
- Bollinger Lower: {indicators.get('bb_lower', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- Volatility: {indicators.get('volatility', 'N/A')}
- Momentum: {indicators.get('momentum', 'N/A')}
- Stochastic %K: {indicators.get('stoch_k', 'N/A')}
- Stochastic %D: {indicators.get('stoch_d', 'N/A')}

INDICATOR SIGNALS:
- Trend: {indicators.get('trend', 'neutral')}
- RSI Signal: {indicators.get('rsi_signal', 'neutral')}
- MACD Signal: {indicators.get('macd_signal_type', 'neutral')}
- Bollinger Band Position: {indicators.get('bb_signal', 'neutral')}

RECENT PRICE ACTION:
{self._format_price_history(price_history)}

Provide your technical analysis following the required JSON format."""

        return prompt

    def _format_price_history(self, price_history: list) -> str:
        """Format price history for display"""
        if not price_history:
            return "No price history available"

        lines = []
        for i, price_data in enumerate(price_history[-10:]):  # Last 10 bars
            if isinstance(price_data, dict):
                lines.append(f"  [{i}] O: {price_data.get('open', 0):.2f}, "
                           f"H: {price_data.get('high', 0):.2f}, "
                           f"L: {price_data.get('low', 0):.2f}, "
                           f"C: {price_data.get('close', 0):.2f}")
            else:
                lines.append(f"  [{i}] {price_data:.2f}")

        return "\n".join(lines)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and generate analysis

        Args:
            context: Dictionary containing:
                - current_price: Current asset price
                - indicators: Technical indicators dict
                - price_history: Recent price data

        Returns:
            Analysis dictionary with trend, momentum, volatility assessment
        """
        response = self.generate_response(context, use_json=True)

        # Validate response structure
        required_fields = ['trend', 'trend_strength', 'momentum', 'volatility', 'confidence', 'summary']
        for field in required_fields:
            if field not in response:
                response[field] = 'unknown' if field != 'confidence' else 0.5

        return response
