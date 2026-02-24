from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderAnalyzer:
    """Fast rule-based sentiment scoring using VADER."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        return {
            "vader_compound": scores["compound"],
            "vader_positive": scores["pos"],
            "vader_negative": scores["neg"],
            "vader_neutral": scores["neu"],
            "vader_label": self._compound_to_label(scores["compound"]),
        }

    @staticmethod
    def _compound_to_label(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"
