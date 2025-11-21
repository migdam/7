"""
News Fetcher
Fetch news from RSS feeds, APIs, and other sources
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json


class NewsFetcher:
    """Base class for news fetchers"""

    def fetch(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Fetch news items"""
        raise NotImplementedError


class RSSNewsFetcher(NewsFetcher):
    """Fetch news from RSS feeds"""

    def __init__(self):
        try:
            import feedparser
            self.feedparser = feedparser
        except ImportError:
            raise ImportError(
                "feedparser not installed. Install with: pip install feedparser"
            )

        # Common financial news RSS feeds
        self.feeds = {
            'reuters': 'https://www.reutersagency.com/feed/',
            'marketwatch': 'https://www.marketwatch.com/rss/',
            'bloomberg': 'https://www.bloomberg.com/feed/podcast/etf-report.xml',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml'
        }

    def fetch(
        self,
        symbol: str = None,
        limit: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch news from RSS feeds

        Args:
            symbol: Stock symbol (for filtering, if supported)
            limit: Maximum number of items per source
            sources: List of source names (default: all)

        Returns:
            List of news items
        """
        if sources is None:
            sources = list(self.feeds.keys())

        news_items = []

        for source in sources:
            if source not in self.feeds:
                continue

            try:
                feed = self.feedparser.parse(self.feeds[source])

                for entry in feed.entries[:limit]:
                    item = {
                        'headline': entry.get('title', ''),
                        'source': source,
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', '')[:200]  # Truncate
                    }
                    news_items.append(item)

            except Exception as e:
                print(f"Error fetching from {source}: {e}")

        return news_items


class NewsAPIFetcher(NewsFetcher):
    """Fetch news from NewsAPI.org"""

    def __init__(self, api_key: str):
        """
        Initialize NewsAPI fetcher

        Args:
            api_key: NewsAPI.org API key
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def fetch(
        self,
        symbol: str = None,
        limit: int = 10,
        query: str = "stock market",
        language: str = "en"
    ) -> List[Dict]:
        """
        Fetch news from NewsAPI

        Args:
            symbol: Stock symbol (added to query)
            limit: Maximum number of items
            query: Search query
            language: Language code

        Returns:
            List of news items
        """
        if symbol:
            query = f"{symbol} {query}"

        url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': language,
            'pageSize': limit,
            'sortBy': 'publishedAt'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            news_items = []
            for article in data.get('articles', []):
                item = {
                    'headline': article.get('title', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'link': article.get('url', ''),
                    'published': article.get('publishedAt', ''),
                    'summary': article.get('description', '')
                }
                news_items.append(item)

            return news_items

        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []


class FinancialModelingPrepFetcher(NewsFetcher):
    """Fetch news from Financial Modeling Prep API"""

    def __init__(self, api_key: str):
        """
        Initialize FMP fetcher

        Args:
            api_key: Financial Modeling Prep API key
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def fetch(
        self,
        symbol: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Fetch stock news from FMP

        Args:
            symbol: Stock symbol
            limit: Maximum number of items

        Returns:
            List of news items
        """
        if not symbol:
            # Get general market news
            url = f"{self.base_url}/stock_news"
            params = {'apikey': self.api_key, 'limit': limit}
        else:
            # Get symbol-specific news
            url = f"{self.base_url}/stock_news"
            params = {'tickers': symbol, 'apikey': self.api_key, 'limit': limit}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            news_items = []
            for article in data[:limit]:
                item = {
                    'headline': article.get('title', ''),
                    'source': article.get('site', 'Unknown'),
                    'link': article.get('url', ''),
                    'published': article.get('publishedDate', ''),
                    'summary': article.get('text', '')[:200]
                }
                news_items.append(item)

            return news_items

        except Exception as e:
            print(f"Error fetching from FMP: {e}")
            return []


class MockNewsFetcher(NewsFetcher):
    """Mock news fetcher for testing"""

    def fetch(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """
        Generate mock news items

        Args:
            symbol: Stock symbol
            limit: Number of items

        Returns:
            List of mock news items
        """
        templates = [
            "{symbol} posts strong earnings, beating analyst expectations",
            "Market analysts upgrade {symbol} to 'Buy' rating",
            "{symbol} announces new product launch for Q3",
            "Regulatory concerns raised for {symbol} by federal agency",
            "{symbol} CEO discusses growth strategy in investor call",
            "Technical analysts see bullish pattern forming in {symbol}",
            "{symbol} reports mixed quarterly results",
            "Insider trading activity detected at {symbol}",
            "Competitor launches product challenging {symbol}'s market share",
            "{symbol} announces strategic partnership with industry leader"
        ]

        sentiments = ['positive', 'positive', 'positive', 'negative', 'neutral',
                     'positive', 'neutral', 'negative', 'negative', 'positive']

        symbol_str = symbol or "MARKET"

        news_items = []
        for i in range(min(limit, len(templates))):
            item = {
                'headline': templates[i].format(symbol=symbol_str),
                'source': 'MockNews',
                'link': f'https://example.com/news/{i}',
                'published': (datetime.now() - timedelta(hours=i)).isoformat(),
                'summary': f'Mock news article {i} about {symbol_str}',
                'sentiment': sentiments[i]
            }
            news_items.append(item)

        return news_items


def get_news(
    symbol: str = None,
    source: str = 'mock',
    limit: int = 10,
    api_key: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Convenience function to fetch news

    Args:
        symbol: Stock symbol
        source: News source ('rss', 'newsapi', 'fmp', 'mock')
        limit: Maximum number of items
        api_key: API key (required for newsapi, fmp)
        **kwargs: Additional source-specific parameters

    Returns:
        List of news items
    """
    if source == 'rss':
        fetcher = RSSNewsFetcher()
    elif source == 'newsapi':
        if not api_key:
            raise ValueError("API key required for NewsAPI")
        fetcher = NewsAPIFetcher(api_key)
    elif source == 'fmp':
        if not api_key:
            raise ValueError("API key required for Financial Modeling Prep")
        fetcher = FinancialModelingPrepFetcher(api_key)
    elif source == 'mock':
        fetcher = MockNewsFetcher()
    else:
        raise ValueError(f"Unknown source: {source}")

    return fetcher.fetch(symbol=symbol, limit=limit, **kwargs)


# Example usage
if __name__ == "__main__":
    print("Fetching mock news...")
    news = get_news(symbol='AAPL', source='mock', limit=5)

    for i, item in enumerate(news, 1):
        print(f"\n{i}. {item['headline']}")
        print(f"   Source: {item['source']}")
        print(f"   Published: {item['published']}")
        if 'sentiment' in item:
            print(f"   Sentiment: {item['sentiment']}")
