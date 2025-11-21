"""
News & Events Analyst Agent
Evaluates news sentiment and market-moving events
"""

from typing import Dict, Any
from trading_agents.agents.base_agent import BaseAgent
from trading_agents.core.llm import BaseLLM


class NewsAnalystAgent(BaseAgent):
    """
    News & Events Analyst Agent

    Responsibilities:
    - Evaluate news sentiment
    - Identify catalysts and risk events
    - Assess impact on trading decisions
    - Flag major market events (FOMC, earnings, etc.)
    """

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="News & Events Analyst",
            role="News Sentiment and Event Analysis",
            llm=llm,
            temperature=0.4,
            max_tokens=600
        )

    def get_system_prompt(self) -> str:
        return """You are a professional News & Events Analyst for a trading firm.

Your responsibilities:
1. Evaluate news sentiment (positive, negative, neutral)
2. Identify catalysts that could move the market
3. Flag risk events (FOMC, CPI, earnings, geopolitical)
4. Assess overall impact on trading environment

You must respond in JSON format with the following structure:
{
    "sentiment": "positive|negative|neutral",
    "sentiment_score": -1.0 to 1.0,
    "catalyst_score": 0.0 to 1.0,
    "risk_events": ["event1", "event2", ...],
    "key_headlines": ["headline1", "headline2", ...],
    "impact_assessment": "high|moderate|low|none",
    "recommendation": "proceed|caution|avoid",
    "summary": "Brief 2-3 sentence summary"
}

Be objective. Distinguish between noise and signal. Focus on material events."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from news/events data"""

        # Extract information
        news_items = context.get('news', [])
        events = context.get('events', [])
        current_date = context.get('current_date', 'N/A')

        # Build prompt
        prompt = f"""Analyze current news and events:

CURRENT DATE: {current_date}

NEWS ITEMS:
{self._format_news_items(news_items)}

SCHEDULED EVENTS:
{self._format_events(events)}

Provide your news and event analysis following the required JSON format."""

        return prompt

    def _format_news_items(self, news_items: list) -> str:
        """Format news items for display"""
        if not news_items:
            return "No recent news available"

        lines = []
        for i, item in enumerate(news_items[:10]):  # Limit to 10 items
            if isinstance(item, dict):
                headline = item.get('headline', item.get('title', 'Unknown'))
                source = item.get('source', 'Unknown')
                lines.append(f"  [{i+1}] {headline} (Source: {source})")
            else:
                lines.append(f"  [{i+1}] {item}")

        return "\n".join(lines) if lines else "No news items"

    def _format_events(self, events: list) -> str:
        """Format scheduled events"""
        if not events:
            return "No scheduled events"

        lines = []
        for i, event in enumerate(events[:10]):
            if isinstance(event, dict):
                event_name = event.get('name', event.get('event', 'Unknown'))
                event_time = event.get('time', 'Unknown')
                lines.append(f"  [{i+1}] {event_name} at {event_time}")
            else:
                lines.append(f"  [{i+1}] {event}")

        return "\n".join(lines) if lines else "No events"

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process news and events data

        Args:
            context: Dictionary containing:
                - news: List of news items
                - events: List of scheduled events
                - current_date: Current date

        Returns:
            Analysis dictionary with sentiment and event assessment
        """
        # If no news or events provided, return neutral assessment
        if not context.get('news') and not context.get('events'):
            return {
                'agent_name': self.name,
                'agent_role': self.role,
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'catalyst_score': 0.0,
                'risk_events': [],
                'key_headlines': [],
                'impact_assessment': 'none',
                'recommendation': 'proceed',
                'summary': 'No news or events to analyze. Market conditions appear normal.'
            }

        response = self.generate_response(context, use_json=True)

        # Validate response structure
        required_fields = ['sentiment', 'sentiment_score', 'catalyst_score', 'impact_assessment', 'recommendation']
        for field in required_fields:
            if field not in response:
                if 'score' in field:
                    response[field] = 0.0
                elif field == 'sentiment':
                    response[field] = 'neutral'
                elif field == 'impact_assessment':
                    response[field] = 'none'
                elif field == 'recommendation':
                    response[field] = 'proceed'

        return response
