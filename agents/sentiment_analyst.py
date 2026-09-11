"""News + Reddit sentiment analysis.

Pulls sentiment signal from allowlisted news domains and subreddits only (see
config/allowed_sources.yaml). Never fetches from arbitrary URLs/subreddits not
present in that allowlist. All external calls must be rate-limited.

TODO:
- Implement Reddit fetch via praw, using credentials from
  config.settings.get_reddit_credentials() and subreddits from the allowlist.
- Implement news fetch restricted to config.settings.load_allowed_sources()["news_domains"].
- Implement sentiment scoring (model/library TBD) and aggregation per symbol.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentRead:
    symbol: str
    reddit_score: float  # -1.0 (bearish) .. 1.0 (bullish)
    news_score: float  # -1.0 (bearish) .. 1.0 (bullish)
    sample_size: int
    notes: str


def fetch_reddit_mentions(symbol: str, subreddits: list[str]) -> list[dict]:
    """Fetch recent mentions of a symbol from the given (allowlisted) subreddits.

    TODO: implement via praw with rate limiting/backoff.
    """
    raise NotImplementedError


def fetch_news_mentions(symbol: str, allowed_domains: list[str]) -> list[dict]:
    """Fetch recent news mentions of a symbol, restricted to allowed_domains.

    TODO: implement with rate limiting/backoff.
    """
    raise NotImplementedError


def score_sentiment(mentions: list[dict]) -> float:
    """Score a list of text mentions on a -1..1 sentiment scale.

    TODO: implement (e.g. via a lexicon or a small model).
    """
    raise NotImplementedError


def analyze(symbol: str) -> SentimentRead:
    """Run the full sentiment pipeline for a symbol.

    TODO: implement, composing the fetch + score functions above, reading the
    allowlist from config.settings.load_allowed_sources().
    """
    raise NotImplementedError
