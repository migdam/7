"""
Sentiment Analyzer
Analyze sentiment of news, social media, and other text data
"""

from typing import Dict, List, Optional
import re


class SentimentAnalyzer:
    """Base sentiment analyzer"""

    def analyze(self, text: str) -> Dict:
        """Analyze text sentiment"""
        raise NotImplementedError


class LexiconBasedAnalyzer(SentimentAnalyzer):
    """Simple lexicon-based sentiment analyzer"""

    def __init__(self):
        # Financial sentiment keywords
        self.positive_words = {
            'bullish', 'rally', 'surge', 'gain', 'profit', 'growth', 'strong',
            'beat', 'exceed', 'outperform', 'upgrade', 'positive', 'rise',
            'jump', 'soar', 'breakout', 'momentum', 'recovery', 'boost',
            'success', 'improve', 'win', 'advance', 'expansion', 'uptrend'
        }

        self.negative_words = {
            'bearish', 'crash', 'plunge', 'loss', 'decline', 'weak', 'miss',
            'underperform', 'downgrade', 'negative', 'fall', 'drop', 'sink',
            'concern', 'risk', 'threat', 'crisis', 'selloff', 'struggle',
            'fail', 'worsen', 'trouble', 'warning', 'retreat', 'downtrend'
        }

        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'significantly': 1.5,
            'substantially': 1.5, 'sharply': 1.5, 'dramatically': 2.0
        }

        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'hardly', 'barely'}

    def analyze(self, text: str) -> Dict:
        """
        Analyze text sentiment using lexicon

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores and classification
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_score = 0
        negative_score = 0

        for i, word in enumerate(words):
            # Check for negation
            negated = False
            if i > 0 and words[i-1] in self.negations:
                negated = True

            # Check for intensifier
            intensifier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier = self.intensifiers[words[i-1]]

            # Score the word
            if word in self.positive_words:
                score = 1.0 * intensifier
                if negated:
                    negative_score += score
                else:
                    positive_score += score

            elif word in self.negative_words:
                score = 1.0 * intensifier
                if negated:
                    positive_score += score
                else:
                    negative_score += score

        # Calculate normalized sentiment
        total_score = positive_score + negative_score
        if total_score > 0:
            sentiment_score = (positive_score - negative_score) / total_score
        else:
            sentiment_score = 0.0

        # Classify sentiment
        if sentiment_score > 0.2:
            sentiment_label = 'positive'
        elif sentiment_score < -0.2:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        return {
            'sentiment': sentiment_label,
            'score': round(sentiment_score, 3),
            'positive_score': round(positive_score, 2),
            'negative_score': round(negative_score, 2),
            'confidence': round(abs(sentiment_score), 3)
        }


class VaderSentimentAnalyzer(SentimentAnalyzer):
    """VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzer"""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "vaderSentiment not installed. Install with: pip install vaderSentiment"
            )

    def analyze(self, text: str) -> Dict:
        """
        Analyze text using VADER

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        scores = self.analyzer.polarity_scores(text)

        # Classify based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': round(compound, 3),
            'positive_score': round(scores['pos'], 3),
            'negative_score': round(scores['neg'], 3),
            'neutral_score': round(scores['neu'], 3),
            'confidence': round(abs(compound), 3)
        }


class TextBlobAnalyzer(SentimentAnalyzer):
    """TextBlob sentiment analyzer"""

    def __init__(self):
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
        except ImportError:
            raise ImportError(
                "textblob not installed. Install with: pip install textblob"
            )

    def analyze(self, text: str) -> Dict:
        """
        Analyze text using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        blob = self.TextBlob(text)
        polarity = blob.sentiment.polarity  # Range: -1 to 1
        subjectivity = blob.sentiment.subjectivity  # Range: 0 to 1

        # Classify
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'confidence': round(abs(polarity), 3)
        }


class AggregatedSentimentAnalyzer:
    """Aggregate results from multiple analyzers"""

    def __init__(self, analyzers: List[SentimentAnalyzer] = None):
        """
        Initialize with multiple analyzers

        Args:
            analyzers: List of analyzer instances
        """
        if analyzers is None:
            # Use lexicon by default (no dependencies)
            self.analyzers = [LexiconBasedAnalyzer()]
        else:
            self.analyzers = analyzers

    def analyze(self, text: str) -> Dict:
        """
        Aggregate sentiment from multiple analyzers

        Args:
            text: Text to analyze

        Returns:
            Aggregated sentiment results
        """
        results = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(text)
                results.append(result)
            except Exception as e:
                print(f"Analyzer failed: {e}")

        if not results:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }

        # Average scores
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)

        # Majority vote for sentiment
        sentiments = [r['sentiment'] for r in results]
        sentiment = max(set(sentiments), key=sentiments.count)

        return {
            'sentiment': sentiment,
            'score': round(avg_score, 3),
            'confidence': round(avg_confidence, 3),
            'analyzer_count': len(results),
            'individual_results': results
        }


def analyze_sentiment(
    text: str,
    method: str = 'lexicon',
    detailed: bool = False
) -> Dict:
    """
    Convenience function for sentiment analysis

    Args:
        text: Text to analyze
        method: Analysis method ('lexicon', 'vader', 'textblob', 'aggregate')
        detailed: Return detailed results

    Returns:
        Sentiment analysis results
    """
    if method == 'lexicon':
        analyzer = LexiconBasedAnalyzer()
    elif method == 'vader':
        analyzer = VaderSentimentAnalyzer()
    elif method == 'textblob':
        analyzer = TextBlobAnalyzer()
    elif method == 'aggregate':
        # Try to use all available analyzers
        analyzers = []
        try:
            analyzers.append(LexiconBasedAnalyzer())
        except:
            pass
        try:
            analyzers.append(VaderSentimentAnalyzer())
        except:
            pass
        try:
            analyzers.append(TextBlobAnalyzer())
        except:
            pass
        analyzer = AggregatedSentimentAnalyzer(analyzers)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = analyzer.analyze(text)

    if not detailed:
        # Return simplified result
        return {
            'sentiment': result['sentiment'],
            'score': result['score']
        }

    return result


def analyze_news_batch(news_items: List[Dict]) -> Dict:
    """
    Analyze sentiment of multiple news items

    Args:
        news_items: List of news dictionaries with 'headline' or 'text'

    Returns:
        Aggregated sentiment analysis
    """
    analyzer = LexiconBasedAnalyzer()

    sentiments = []
    scores = []

    for item in news_items:
        text = item.get('headline', item.get('text', ''))
        if text:
            result = analyzer.analyze(text)
            sentiments.append(result['sentiment'])
            scores.append(result['score'])

    if not scores:
        return {
            'overall_sentiment': 'neutral',
            'average_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }

    # Calculate aggregates
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')

    avg_score = sum(scores) / len(scores)

    # Overall sentiment
    if positive_count > negative_count:
        overall = 'positive'
    elif negative_count > positive_count:
        overall = 'negative'
    else:
        overall = 'neutral'

    return {
        'overall_sentiment': overall,
        'average_score': round(avg_score, 3),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_items': len(news_items),
        'sentiment_ratio': round(positive_count / len(news_items), 2) if news_items else 0
    }


# Example usage
if __name__ == "__main__":
    texts = [
        "Stock surges on strong earnings beat, investors bullish",
        "Company reports significant losses, shares plunge",
        "Market remains stable with mixed trading activity"
    ]

    print("Sentiment Analysis Examples")
    print("=" * 60)

    for i, text in enumerate(texts, 1):
        result = analyze_sentiment(text, method='lexicon', detailed=True)
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Score: {result['score']}")
        print(f"   Confidence: {result['confidence']}")
