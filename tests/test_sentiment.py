"""Tests for trading_agents.enhancements.sentiment_analyzer"""

import pytest
from trading_agents.enhancements.sentiment_analyzer import (
    LexiconBasedAnalyzer,
    AggregatedSentimentAnalyzer,
    analyze_sentiment,
    analyze_news_batch
)


class TestLexiconBasedAnalyzer:

    def test_positive_text(self):
        analyzer = LexiconBasedAnalyzer()
        result = analyzer.analyze("Stock surges on strong earnings beat, investors bullish")
        assert result['sentiment'] == 'positive'
        assert result['score'] > 0

    def test_negative_text(self):
        analyzer = LexiconBasedAnalyzer()
        result = analyzer.analyze("Company reports significant losses, shares plunge")
        assert result['sentiment'] == 'negative'
        assert result['score'] < 0

    def test_neutral_text(self):
        analyzer = LexiconBasedAnalyzer()
        result = analyzer.analyze("The weather is nice today")
        assert result['sentiment'] == 'neutral'

    def test_consistent_return_keys(self):
        analyzer = LexiconBasedAnalyzer()
        result = analyzer.analyze("Stock surges on strong earnings")
        expected_keys = {'sentiment', 'score', 'positive_score', 'negative_score', 'neutral_score', 'confidence'}
        assert expected_keys.issubset(set(result.keys()))

    def test_negation_handling(self):
        analyzer = LexiconBasedAnalyzer()
        result = analyzer.analyze("not bullish")
        # "not bullish" should flip to negative
        assert result['negative_score'] > 0 or result['score'] <= 0


class TestAggregatedSentimentAnalyzer:

    def test_default_uses_lexicon(self):
        analyzer = AggregatedSentimentAnalyzer()
        result = analyzer.analyze("Stock rally boosts profits")
        assert 'sentiment' in result
        assert 'score' in result

    def test_empty_analyzers(self):
        analyzer = AggregatedSentimentAnalyzer(analyzers=[])
        result = analyzer.analyze("Some text")
        assert result['sentiment'] == 'neutral'


class TestAnalyzeSentiment:

    def test_lexicon_method(self):
        result = analyze_sentiment("bullish rally growth", method='lexicon')
        assert 'sentiment' in result
        assert 'score' in result

    def test_detailed_flag(self):
        result = analyze_sentiment("bullish rally growth", method='lexicon', detailed=True)
        assert 'confidence' in result

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            analyze_sentiment("text", method='nonexistent')


class TestAnalyzeNewsBatch:

    def test_batch_analysis(self):
        news = [
            {'headline': 'Stock surges on earnings beat'},
            {'headline': 'Market crash fears grow'},
            {'headline': 'Normal trading activity today'}
        ]
        result = analyze_news_batch(news)
        assert 'overall_sentiment' in result
        assert 'average_score' in result
        assert result['total_items'] == 3
        assert result['positive_count'] + result['negative_count'] + result['neutral_count'] == 3

    def test_empty_batch(self):
        result = analyze_news_batch([])
        assert result['overall_sentiment'] == 'neutral'
        assert result['average_score'] == 0.0
