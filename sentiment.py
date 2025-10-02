from __future__ import annotations
import functools
from typing import Dict

try:
    from transformers import pipeline
    _finbert = pipeline(
        task="sentiment-analysis",
        model="ProsusAI/finbert",
        return_all_scores=True
    )
    HAS_FINBERT = True
except Exception:
    _finbert = None
    HAS_FINBERT = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:
    _vader = None
    HAS_VADER = False

LABEL_ORDER = ["negative", "neutral", "positive"]

@functools.lru_cache(maxsize=2048)
def finbert_scores(text: str) -> Dict[str, float]:
    if HAS_FINBERT:
        scores = _finbert(text[:512])[0]
        return {s['label'].lower(): float(s['score']) for s in scores}
    elif HAS_VADER:
        vs = _vader.polarity_scores(text)
        comp = vs['compound']
        pos, neg, neu = max(0.0, comp), max(0.0, -comp), vs['neu']
        s = pos + neg + neu + 1e-9
        return {"positive": pos/s, "negative": neg/s, "neutral": neu/s}
    return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

def label_from_scores(scores: Dict[str, float]) -> str:
    return max(LABEL_ORDER, key=lambda k: scores.get(k, 0.0))

def score_scalar(scores: Dict[str, float]) -> float:
    return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
