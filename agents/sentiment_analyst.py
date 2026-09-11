"""News + Reddit sentiment analysis.

Pulls sentiment signal from allowlisted news domains and subreddits only (see
config/allowed_sources.yaml). Never fetches from arbitrary URLs or subreddits
not present in that allowlist. All external calls are rate-limited.

Sentiment is scored with a finance-tuned lexicon rather than a general-purpose
model: in market text the word that matters is "beat", "miss", "guidance" or
"downgrade", and a general sentiment model reads those as neutral. The lexicon
is deliberately small and inspectable -- when a score looks wrong you can see
exactly which words produced it.

News is scored the way a desk reads a feed rather than as a flat average: each
article is weighted by age (see agents/news_signal.py), and the *volume* of
coverage is tracked separately from its direction, because a sudden burst of
articles is a signal in itself. Retrieval lives in agents/news_sources.py, which
prefers Tiingo (dated, detailed) and falls back to yfinance headlines.

Reddit and Tiingo both require API credentials. When they are absent those legs
are skipped and reported as unavailable rather than raising: a missing optional
source should degrade the read, not break the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents import news_signal, news_sources
from common.rate_limit import get_limiter
from config.settings import get_reddit_credentials, load_allowed_sources

# Finance-tuned sentiment lexicon. Weights are in [-1, 1].
_POSITIVE_TERMS = {
    "beat": 0.8, "beats": 0.8, "outperform": 0.8, "upgrade": 0.9, "upgraded": 0.9,
    "surge": 0.7, "surged": 0.7, "rally": 0.6, "rallied": 0.6, "soar": 0.8,
    "soared": 0.8, "jump": 0.5, "jumped": 0.5, "gain": 0.4, "gains": 0.4,
    "profit": 0.5, "profitable": 0.6, "growth": 0.5, "record": 0.6, "strong": 0.6,
    "bullish": 0.9, "buy": 0.5, "breakout": 0.6, "momentum": 0.4, "raises": 0.6,
    "raised": 0.5, "expansion": 0.4, "dividend": 0.3, "buyback": 0.6, "win": 0.5,
    "wins": 0.5, "approval": 0.6, "approved": 0.6, "optimistic": 0.6, "top": 0.4,
    "tops": 0.6, "exceeded": 0.7, "positive": 0.5, "rebound": 0.6, "recovery": 0.5,
}

_NEGATIVE_TERMS = {
    "miss": -0.8, "missed": -0.8, "misses": -0.8, "downgrade": -0.9,
    "downgraded": -0.9, "plunge": -0.8, "plunged": -0.8, "crash": -0.9,
    "crashed": -0.9, "tumble": -0.7, "tumbled": -0.7, "slump": -0.6,
    "fall": -0.4, "falls": -0.4, "fell": -0.4, "drop": -0.5, "dropped": -0.5,
    "loss": -0.6, "losses": -0.6, "weak": -0.6, "bearish": -0.9, "sell": -0.5,
    "selloff": -0.8, "lawsuit": -0.6, "probe": -0.6, "investigation": -0.6,
    "fraud": -0.9, "bankruptcy": -1.0, "layoff": -0.6, "layoffs": -0.6,
    "cuts": -0.5, "cut": -0.4, "warning": -0.7, "warns": -0.7, "risk": -0.3,
    "concern": -0.4, "concerns": -0.4, "decline": -0.5, "declined": -0.5,
    "slowdown": -0.6, "negative": -0.5, "disappointing": -0.7, "halt": -0.6,
    "recall": -0.6, "delay": -0.4, "delayed": -0.4, "overvalued": -0.6,
}

_LEXICON = {**_POSITIVE_TERMS, **_NEGATIVE_TERMS}

# Negators flip the polarity of the term that follows them within this window.
_NEGATORS = frozenset({"not", "no", "never", "without", "isn't", "wasn't", "won't", "cannot"})
_NEGATION_WINDOW = 3

# Suffixes stripped when a token is not in the lexicon verbatim, so "surges"
# finds "surge" without every inflection needing its own entry. Longest first,
# so "ing" is tried before "s" on a word like "surging".
_SUFFIXES = ("ings", "ing", "ers", "er", "ed", "es", "s")

DEFAULT_POST_LIMIT = 50


@dataclass(frozen=True)
class SentimentRead:
    symbol: str
    reddit_score: float  # -1.0 (bearish) .. 1.0 (bullish)
    news_score: float  # -1.0 (bearish) .. 1.0 (bullish)
    sample_size: int
    notes: str
    signal: str = "neutral"  # "bullish" / "bearish" / "neutral" / "unknown"
    reddit_sample: int = 0
    news_sample: int = 0
    sources_unavailable: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    news_detail: news_signal.NewsSignal | None = None
    news_provider: str = ""


def _lookup(token: str) -> float | None:
    """Find a token's weight, falling back to a suffix-stripped stem.

    Only strips when the result is itself a lexicon entry, so "loss" is never
    mangled into "los" and unrelated words cannot accidentally match.
    """
    weight = _LEXICON.get(token)
    if weight is not None:
        return weight

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            weight = _LEXICON.get(stem)
            if weight is not None:
                return weight
            # "surged" -> "surge": restore a dropped trailing "e".
            weight = _LEXICON.get(stem + "e")
            if weight is not None:
                return weight
    return None


def score_text(text: str) -> float:
    """Score a single piece of text on a -1..1 sentiment scale.

    Returns 0.0 for text containing no lexicon terms, which is distinct from
    text whose positive and negative terms cancel out -- both read neutral, but
    only the latter is evidence of anything.
    """
    tokens = re.findall(r"[a-z']+", (text or "").lower())
    if not tokens:
        return 0.0

    total = 0.0
    hits = 0

    for index, token in enumerate(tokens):
        weight = _lookup(token)
        if weight is None:
            continue
        # Look back a few tokens for a negator ("not a beat" is not bullish).
        window = tokens[max(0, index - _NEGATION_WINDOW):index]
        if any(w in _NEGATORS for w in window):
            weight = -weight
        total += weight
        hits += 1

    if hits == 0:
        return 0.0

    # Average rather than sum, so a long article isn't automatically extreme.
    return max(-1.0, min(1.0, total / hits))


def score_sentiment(mentions: list[dict]) -> float:
    """Score a list of text mentions on a -1..1 scale.

    Mentions are weighted by their `weight` key when present (Reddit uses post
    score, so a heavily upvoted post counts for more than a lone comment).
    """
    if not mentions:
        return 0.0

    weighted_total = 0.0
    weight_total = 0.0

    for mention in mentions:
        text = f"{mention.get('title', '')} {mention.get('body', '')}".strip()
        score = score_text(text)
        # Dampen the weight so one viral post cannot dominate the read.
        weight = 1.0 + min(float(mention.get("weight", 0) or 0), 1000) / 500.0
        weighted_total += score * weight
        weight_total += weight

    if weight_total == 0:
        return 0.0
    return max(-1.0, min(1.0, weighted_total / weight_total))


def fetch_reddit_mentions(
    symbol: str, subreddits: list[str], limit: int = DEFAULT_POST_LIMIT
) -> list[dict]:
    """Fetch recent mentions of a symbol from the given (allowlisted) subreddits.

    Raises ValueError if a subreddit is not on the allowlist, or if credentials
    are missing -- the caller decides whether that is fatal.
    """
    allowed = set(load_allowed_sources().get("subreddits") or [])
    requested = [s for s in subreddits if s]

    disallowed = [s for s in requested if s not in allowed]
    if disallowed:
        raise ValueError(
            f"Subreddits not on the allowlist: {disallowed}. Add them to "
            f"config/allowed_sources.yaml first (see SECURITY.md rule 3)."
        )

    credentials = get_reddit_credentials()  # raises if unset

    import praw  # imported lazily so the module loads without praw configured

    reddit = praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        user_agent=credentials.user_agent,
        check_for_async=False,
    )

    mentions: list[dict] = []
    query = f"{symbol}"

    for subreddit in requested:
        get_limiter("reddit").wait()
        try:
            results = reddit.subreddit(subreddit).search(
                query, sort="new", time_filter="month", limit=limit
            )
            for post in results:
                mentions.append(
                    {
                        "title": post.title,
                        "body": (post.selftext or "")[:2000],
                        "weight": getattr(post, "score", 0),
                        "source": f"r/{subreddit}",
                    }
                )
        except Exception:
            # One failing subreddit shouldn't discard the others.
            continue

    return mentions


def fetch_news_mentions(symbol: str, allowed_domains: list[str]) -> list[news_sources.NewsItem]:
    """Fetch recent news for a symbol, restricted to allowed_domains.

    Delegates to news_sources, which prefers Tiingo and falls back to yfinance.
    Kept as a named function because it is part of this module's public surface.
    """
    items, _ = news_sources.fetch_news(symbol, allowed_domains)
    return items


def analyze(
    symbol: str,
    strict_allowlist: bool = True,
    half_life_days: float = news_signal.DEFAULT_HALF_LIFE_DAYS,
) -> SentimentRead:
    """Run the full sentiment pipeline for a symbol.

    News is recency-weighted and its coverage volume is measured separately;
    Reddit is scored by upvote-weighted average. Sources that are unavailable
    (no credentials, no matching articles) are reported in `sources_unavailable`
    rather than raising.

    With strict_allowlist=False, news from outside the allowlisted domains is
    also counted. That is off by default because SECURITY.md rule 3 requires
    allowlisted sources; enable it knowingly.
    """
    symbol = symbol.strip().upper()
    config = load_allowed_sources()
    subreddits = list(config.get("subreddits") or [])
    news_domains = list(config.get("news_domains") or [])

    unavailable: list[str] = []
    reasons: list[str] = []

    # --- Reddit -----------------------------------------------------------
    reddit_mentions: list[dict] = []
    try:
        reddit_mentions = fetch_reddit_mentions(symbol, subreddits)
    except ValueError as exc:
        unavailable.append(f"reddit ({exc.args[0].split('.')[0]})")
    except Exception as exc:
        unavailable.append(f"reddit ({type(exc).__name__})")

    reddit_score = score_sentiment(reddit_mentions)

    # --- News -------------------------------------------------------------
    news_items, news_unavailable = news_sources.fetch_news(
        symbol, news_domains if strict_allowlist else []
    )
    unavailable.extend(news_unavailable)

    detail = news_signal.build_news_signal(
        news_items, score_text, half_life_days=half_life_days
    )
    news_score = detail.score
    provider = news_items[0].provider if news_items else ""

    # --- Combine ----------------------------------------------------------
    sample_size = len(reddit_mentions) + len(news_items)

    if sample_size == 0:
        hints = []
        if any(u.startswith("tiingo") for u in unavailable):
            hints.append(
                "Tiingo needs TIINGO_API_KEY in .env (free key at tiingo.com) "
                "for dated news with volume tracking."
            )
        if any(u.startswith("reddit") for u in unavailable):
            hints.append(
                "Reddit needs REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and "
                "REDDIT_USER_AGENT in .env to contribute."
            )
        return SentimentRead(
            symbol=symbol,
            reddit_score=0.0,
            news_score=0.0,
            sample_size=0,
            signal="unknown",
            sources_unavailable=unavailable,
            news_detail=detail,
            notes=(
                f"No sentiment data available for {symbol}. "
                f"Unavailable: {', '.join(unavailable) or 'none'}."
            ),
            reasons=hints,
        )

    # Weight each leg by how much evidence it actually carries.
    combined = (
        reddit_score * len(reddit_mentions) + news_score * len(news_items)
    ) / sample_size

    if combined > 0.15:
        signal = "bullish"
    elif combined < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    if reddit_mentions:
        reasons.append(
            f"Reddit: {reddit_score:+.2f} across {len(reddit_mentions)} posts."
        )
    reasons.extend(detail.reasons)
    if unavailable:
        # Keep the reason line readable: name the providers, not their full
        # error text, which for a missing key is a multi-sentence hint.
        names = [entry.split(" (", 1)[0] for entry in unavailable]
        reasons.append(f"Unavailable sources: {', '.join(names)}.")

    notes = (
        f"Sentiment for {symbol} reads {signal} ({combined:+.2f}) "
        f"from {sample_size} mentions"
    )
    notes += f" via {provider}." if provider else "."
    if detail.volume.status in ("spike", "quiet"):
        notes += f" Coverage volume is a {detail.volume.status}."

    return SentimentRead(
        symbol=symbol,
        reddit_score=reddit_score,
        news_score=news_score,
        sample_size=sample_size,
        signal=signal,
        reddit_sample=len(reddit_mentions),
        news_sample=len(news_items),
        sources_unavailable=unavailable,
        notes=notes,
        reasons=reasons,
        news_detail=detail,
        news_provider=provider,
    )
