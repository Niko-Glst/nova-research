"""News retrieval, normalized across providers.

Every provider returns a different shape; this module flattens them into one
`NewsItem` so the sentiment analyst never has to care where an article came
from. The important field is `published`: without a timestamp you cannot weight
recency or detect a change in coverage volume, which is most of what makes a
news signal worth anything.

Providers, in the order they are tried:

- Finnhub -- needs FINNHUB_API_KEY. Company news over an explicit date range on
  the free tier, which is what makes recency weighting and volume tracking work.
- Tiingo -- needs TIINGO_API_KEY. Good data, but its News API is a paid add-on:
  a free key returns HTTP 403 here ("You do not have permission to access the
  News API") even though the same key works for prices and fundamentals. Kept
  because it is the better feed once enabled.
- yfinance -- no credentials, but returns only the newest ten headlines. Enough
  for a sentiment read, not enough to measure coverage volume (see
  TRUNCATED_FEED_PROVIDERS below).

Sources are still filtered against the allowlist in config/allowed_sources.yaml
(SECURITY.md rule 3).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from common.rate_limit import get_limiter
from config.settings import get_finnhub_credentials, get_tiingo_credentials

TIINGO_NEWS_URL = "https://api.tiingo.com/tiingo/news"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_REQUEST_TIMEOUT_SECONDS = 15

# How many articles to request from Tiingo per symbol. Enough to measure a
# volume trend over several weeks without pulling an unbounded history.
TIINGO_DEFAULT_LIMIT = 100


# Providers that return only the newest handful of articles, with no way to ask
# for a historical window. Coverage volume cannot be measured from these: the
# feed is truncated by construction, so *every* symbol looks like a burst with
# no baseline behind it. See news_signal.analyze_volume.
TRUNCATED_FEED_PROVIDERS = frozenset({"yfinance"})


@dataclass(frozen=True)
class NewsItem:
    """One article, normalized across providers."""

    title: str
    body: str
    published: datetime | None  # UTC; None when the provider omitted it
    source: str
    url: str
    provider: str  # which backend produced this item

    @property
    def supports_volume_analysis(self) -> bool:
        """Whether this item's provider returns a window rather than a snapshot."""
        return self.provider not in TRUNCATED_FEED_PROVIDERS

    @property
    def age_days(self) -> float | None:
        """Days since publication, or None when the date is unknown."""
        if self.published is None:
            return None
        delta = datetime.now(timezone.utc) - self.published
        return max(0.0, delta.total_seconds() / 86400.0)


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _matches_allowlist(haystack: str, allowed_domains: list[str]) -> bool:
    """Whether a source string matches any allowlisted domain.

    An empty allowlist means "do not filter" -- the caller has opted out
    deliberately (see sentiment_analyst.analyze's strict_allowlist flag).
    """
    if not allowed_domains:
        return True
    lowered = haystack.lower()
    return any(domain.lower() in lowered for domain in allowed_domains)


def fetch_tiingo_news(
    symbol: str,
    allowed_domains: list[str],
    lookback_days: int = 30,
    limit: int = TIINGO_DEFAULT_LIMIT,
) -> list[NewsItem]:
    """Fetch news for a symbol from Tiingo.

    Raises ValueError when the API key is absent or the request fails, so the
    caller can fall back to another provider.
    """
    credentials = get_tiingo_credentials()  # raises when TIINGO_API_KEY is unset

    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    query = urllib.parse.urlencode(
        {
            "tickers": symbol.strip().upper(),
            "startDate": start.isoformat(),
            "limit": str(limit),
            "sortBy": "publishedDate",
        }
    )
    request = urllib.request.Request(
        f"{TIINGO_NEWS_URL}?{query}",
        headers={
            "Content-Type": "application/json",
            # Tiingo accepts the key as a header token, which keeps it out of
            # the URL (and therefore out of logs and error messages).
            "Authorization": f"Token {credentials.api_key}",
        },
    )

    get_limiter("tiingo").wait()
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Deliberately does not echo the response body: Tiingo includes the
        # supplied token in some error payloads.
        raise ValueError(f"Tiingo request failed with HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"Tiingo request failed: {exc.reason}") from None
    except json.JSONDecodeError:
        raise ValueError("Tiingo returned a response that was not valid JSON.") from None

    if not isinstance(payload, list):
        raise ValueError("Tiingo returned an unexpected payload shape.")

    items: list[NewsItem] = []
    for article in payload:
        if not isinstance(article, dict):
            continue
        source = str(article.get("source") or "")
        url = str(article.get("url") or "")
        if not _matches_allowlist(f"{url} {source}", allowed_domains):
            continue
        items.append(
            NewsItem(
                title=str(article.get("title") or ""),
                body=str(article.get("description") or "")[:2000],
                published=_parse_timestamp(article.get("publishedDate")),
                source=source,
                url=url,
                provider="tiingo",
            )
        )
    return items


def fetch_finnhub_news(
    symbol: str,
    allowed_domains: list[str],
    lookback_days: int = 30,
    end_days_ago: int = 0,
) -> list[NewsItem]:
    """Fetch company news for a symbol from Finnhub over one date range.

    Finnhub's company-news endpoint takes an explicit date range and is
    available on the free tier, which is what makes volume tracking possible.
    It does cap the response at a few hundred articles, so `end_days_ago` lets
    a caller slice the history into windows that each fit under that cap --
    see fetch_finnhub_windowed.

    Raises ValueError when the API key is absent or the request fails, so the
    caller can fall back to another provider.
    """
    credentials = get_finnhub_credentials()  # raises when FINNHUB_API_KEY is unset

    end = datetime.now(timezone.utc).date() - timedelta(days=end_days_ago)
    start = end - timedelta(days=lookback_days)
    query = urllib.parse.urlencode(
        {
            "symbol": symbol.strip().upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
        }
    )
    request = urllib.request.Request(
        f"{FINNHUB_NEWS_URL}?{query}",
        # Finnhub accepts the key as a header, keeping it out of the URL and
        # therefore out of logs and error messages.
        headers={"X-Finnhub-Token": credentials.api_key},
    )

    get_limiter("finnhub").wait()
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Finnhub request failed with HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"Finnhub request failed: {exc.reason}") from None
    except json.JSONDecodeError:
        raise ValueError("Finnhub returned a response that was not valid JSON.") from None

    if not isinstance(payload, list):
        raise ValueError("Finnhub returned an unexpected payload shape.")

    items: list[NewsItem] = []
    for article in payload:
        if not isinstance(article, dict):
            continue
        source = str(article.get("source") or "")
        url = str(article.get("url") or "")
        if not _matches_allowlist(f"{url} {source}", allowed_domains):
            continue

        # Finnhub timestamps are Unix epoch seconds.
        published = None
        epoch = article.get("datetime")
        if isinstance(epoch, (int, float)) and epoch > 0:
            published = datetime.fromtimestamp(epoch, tz=timezone.utc)

        items.append(
            NewsItem(
                title=str(article.get("headline") or ""),
                body=str(article.get("summary") or "")[:2000],
                published=published,
                source=source,
                url=url,
                provider="finnhub",
            )
        )
    return items


def fetch_finnhub_windowed(
    symbol: str,
    allowed_domains: list[str],
    lookback_days: int = 30,
    recent_window_days: int = 3,
) -> list[NewsItem]:
    """Fetch Finnhub news as two windows so the per-response cap cannot hide the baseline.

    A single 30-day request for a heavily covered name comes back capped at a
    few hundred articles, all from the last day or two -- which makes the older
    baseline look empty and every such name look like a spike. Requesting the
    recent window and the baseline window separately gives each its own cap, so
    both periods are represented.

    Falls back to a single request if the second window fails, since a partial
    result still supports sentiment scoring.
    """
    recent = fetch_finnhub_news(
        symbol, allowed_domains, lookback_days=recent_window_days, end_days_ago=0
    )

    baseline_span = max(1, lookback_days - recent_window_days)
    try:
        baseline = fetch_finnhub_news(
            symbol,
            allowed_domains,
            lookback_days=baseline_span,
            end_days_ago=recent_window_days,
        )
    except ValueError:
        baseline = []

    # De-duplicate on URL: the windows share a boundary day, so an article can
    # legitimately appear in both responses.
    seen: set[str] = set()
    merged: list[NewsItem] = []
    for item in recent + baseline:
        key = item.url or f"{item.title}|{item.published}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def fetch_yfinance_news(symbol: str, allowed_domains: list[str]) -> list[NewsItem]:
    """Fetch headlines yfinance attaches to a ticker.

    No credentials required, but the feed is short and its timestamps are
    inconsistent across yfinance versions.
    """
    import yfinance as yf

    get_limiter("yfinance").wait()
    try:
        articles = yf.Ticker(symbol.strip().upper()).news or []
    except Exception:
        return []

    items: list[NewsItem] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        # yfinance has moved this payload between a flat and a nested shape.
        content = article.get("content")
        record = content if isinstance(content, dict) else article
        if not isinstance(record, dict):
            continue

        provider_field = record.get("provider")
        publisher = (
            provider_field.get("displayName")
            if isinstance(provider_field, dict)
            else record.get("publisher") or ""
        )

        url_field = record.get("canonicalUrl") or record.get("clickThroughUrl")
        url = (
            url_field.get("url")
            if isinstance(url_field, dict)
            else record.get("link") or ""
        )

        if not _matches_allowlist(f"{url} {publisher}", allowed_domains):
            continue

        published = _parse_timestamp(record.get("pubDate") or record.get("displayTime"))
        if published is None:
            epoch = record.get("providerPublishTime")
            if isinstance(epoch, (int, float)):
                published = datetime.fromtimestamp(epoch, tz=timezone.utc)

        items.append(
            NewsItem(
                title=str(record.get("title") or ""),
                body=str(record.get("summary") or record.get("description") or "")[:2000],
                published=published,
                source=str(publisher or ""),
                url=str(url or ""),
                provider="yfinance",
            )
        )
    return items


def fetch_news(
    symbol: str, allowed_domains: list[str], lookback_days: int = 30
) -> tuple[list[NewsItem], list[str]]:
    """Fetch news from the best available provider.

    Returns (items, unavailable) where `unavailable` describes each provider
    that could not be used, so the caller can report why a source is missing
    rather than silently returning less data.
    """
    unavailable: list[str] = []

    # Dated providers first: only they can support recency weighting and volume.
    for name, fetch in (
        ("finnhub", fetch_finnhub_windowed),
        ("tiingo", fetch_tiingo_news),
    ):
        try:
            items = fetch(symbol, allowed_domains, lookback_days=lookback_days)
        except ValueError as exc:
            unavailable.append(f"{name} ({exc})")
            continue
        if items:
            return items, unavailable
        unavailable.append(f"{name} (no articles matched the allowlist)")

    items = fetch_yfinance_news(symbol, allowed_domains)
    if not items:
        unavailable.append("yfinance (no headlines matched the allowlist)")
    return items, unavailable
