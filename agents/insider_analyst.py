"""Insider transaction analysis (Form 4 activity, via yfinance).

Insiders sell for many reasons -- diversification, taxes, a scheduled 10b5-1
plan -- but they buy on the open market for essentially one. That asymmetry is
the signal this module tries to isolate.

The central problem with raw insider data is that most of it is compensation,
not conviction: a "Stock Award(Grant)" at a price of 0.00 is payroll, and
counting it as a purchase would make every company look like its executives are
loading up. `classify_transaction` separates those categories so the score
reflects only open-market decisions.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from common.rate_limit import get_limiter

# Transaction categories. Only OPEN_MARKET_BUY and OPEN_MARKET_SELL feed the score.
BUY = "open_market_buy"
SELL = "open_market_sell"
GRANT = "grant"  # compensation: awards, vesting, RSU delivery
OPTION = "option_exercise"  # exercising options, often followed by a sale
OTHER = "other"

# yfinance exposes the real nature of a transaction in its free-text `Text`
# column; the `Transaction` column is frequently empty. These patterns read it.
_GRANT_PATTERNS = (
    r"\bstock award\b",
    r"\bgrant\b",
    r"\baward\b",
    r"\bvest",
    r"\brestricted stock\b",
    r"\brsu\b",
)
_OPTION_PATTERNS = (r"\bexercise\b", r"\bconversion\b", r"\boption\b")
_BUY_PATTERNS = (r"\bpurchase\b", r"\bbuy\b", r"\bacquisition\b", r"\bbought\b")
_SELL_PATTERNS = (r"\bsale\b", r"\bsell\b", r"\bsold\b", r"\bdisposition\b")

# Roles whose trades carry more weight: a CEO or CFO sees the whole picture,
# where a director may not.
_SENIOR_ROLE_PATTERNS = (
    r"chief executive",
    r"\bceo\b",
    r"chief financial",
    r"\bcfo\b",
    r"chief operating",
    r"\bcoo\b",
    r"president",
    r"chairman",
)

DEFAULT_LOOKBACK_DAYS = 180

# A net buy/sell ratio this far from neutral counts as a real skew.
_SIGNAL_THRESHOLD = 0.20


@dataclass(frozen=True)
class InsiderTransaction:
    """One classified Form 4 filing."""

    insider: str
    position: str
    date: datetime | None
    shares: float
    value: float
    category: str
    is_senior: bool
    description: str


@dataclass(frozen=True)
class InsiderRead:
    symbol: str
    signal: str  # "bullish" / "bearish" / "neutral" / "unknown"
    lookback_days: int
    buy_count: int
    sell_count: int
    buy_value: float
    sell_value: float
    net_value: float
    senior_buy_count: int
    senior_sell_count: int
    transactions: list[InsiderTransaction] = field(default_factory=list)
    notes: str = ""
    reasons: list[str] = field(default_factory=list)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_transaction(text: str, value: float) -> str:
    """Classify a Form 4 line into one of the module's categories.

    Order matters: grants and option exercises are checked before buy/sell,
    because their descriptions often contain acquisition language too
    ("Stock Award(Grant)" is an acquisition of shares, but not a purchase).
    """
    lowered = (text or "").lower()

    if _matches_any(lowered, _GRANT_PATTERNS):
        return GRANT
    if _matches_any(lowered, _OPTION_PATTERNS):
        return OPTION
    if _matches_any(lowered, _SELL_PATTERNS):
        return SELL
    if _matches_any(lowered, _BUY_PATTERNS):
        # A zero-value "purchase" is not an open-market buy -- it is almost
        # always a grant whose description didn't match the patterns above.
        return BUY if value > 0 else GRANT
    return OTHER


def _is_senior(position: str) -> bool:
    return _matches_any((position or "").lower(), _SENIOR_ROLE_PATTERNS)


def fetch_insider_transactions(symbol: str) -> pd.DataFrame:
    """Fetch raw insider transactions for a symbol, rate-limited.

    Returns an empty DataFrame when the provider has no filings for the symbol,
    which is a normal outcome rather than an error.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    get_limiter("yfinance").wait()
    try:
        transactions = yf.Ticker(symbol.strip().upper()).insider_transactions
    except Exception as exc:
        raise ValueError(
            f"Could not fetch insider transactions for {symbol!r}: {exc}"
        ) from exc

    if transactions is None:
        return pd.DataFrame()
    return transactions


def _parse_rows(frame: pd.DataFrame, cutoff: datetime) -> list[InsiderTransaction]:
    """Turn raw provider rows into classified transactions within the window."""
    parsed: list[InsiderTransaction] = []

    for _, row in frame.iterrows():
        raw_date = row.get("Start Date")
        date: datetime | None = None
        if raw_date is not None and not pd.isna(raw_date):
            stamp = pd.to_datetime(raw_date, errors="coerce")
            if not pd.isna(stamp):
                # Normalize to UTC so the cutoff comparison is well defined
                # regardless of whether the provider tagged a timezone.
                date = stamp.tz_localize(timezone.utc) if stamp.tzinfo is None else stamp
                date = date.to_pydatetime()

        if date is not None and date < cutoff:
            continue

        def _number(key: str) -> float:
            raw = row.get(key)
            if raw is None or pd.isna(raw):
                return 0.0
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        description = str(row.get("Text") or "")
        value = _number("Value")
        position = str(row.get("Position") or "")

        parsed.append(
            InsiderTransaction(
                insider=str(row.get("Insider") or "unknown"),
                position=position,
                date=date,
                shares=_number("Shares"),
                value=value,
                category=classify_transaction(description, value),
                is_senior=_is_senior(position),
                description=description,
            )
        )

    return parsed


def analyze(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> InsiderRead:
    """Analyze recent insider activity for a symbol.

    Scores only open-market buys and sells; grants and option exercises are
    parsed and returned but excluded from the signal.
    """
    symbol = symbol.strip().upper()
    frame = fetch_insider_transactions(symbol)

    if frame.empty:
        return InsiderRead(
            symbol=symbol,
            signal="unknown",
            lookback_days=lookback_days,
            buy_count=0,
            sell_count=0,
            buy_value=0.0,
            sell_value=0.0,
            net_value=0.0,
            senior_buy_count=0,
            senior_sell_count=0,
            notes=f"No insider transactions reported for {symbol}.",
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    transactions = _parse_rows(frame, cutoff)

    buys = [t for t in transactions if t.category == BUY]
    sells = [t for t in transactions if t.category == SELL]
    grants = [t for t in transactions if t.category == GRANT]

    buy_value = sum(t.value for t in buys)
    sell_value = sum(t.value for t in sells)
    net_value = buy_value - sell_value
    total_value = buy_value + sell_value

    senior_buys = [t for t in buys if t.is_senior]
    senior_sells = [t for t in sells if t.is_senior]

    reasons: list[str] = []
    if not buys and not sells:
        signal = "neutral"
        reasons.append(
            f"No open-market buys or sells in the last {lookback_days} days "
            f"({len(grants)} compensation grants were filed, which carry no signal)."
        )
    else:
        # Ratio in [-1, 1]: +1 is all buying, -1 is all selling.
        ratio = net_value / total_value if total_value else 0.0

        if ratio > _SIGNAL_THRESHOLD:
            signal = "bullish"
        elif ratio < -_SIGNAL_THRESHOLD:
            signal = "bearish"
        else:
            signal = "neutral"

        reasons.append(
            f"{len(buys)} open-market buys (${buy_value:,.0f}) versus "
            f"{len(sells)} sells (${sell_value:,.0f}) over {lookback_days} days."
        )
        reasons.append(f"Net insider flow: ${net_value:,.0f} ({ratio:+.0%} skew).")

        if senior_buys:
            reasons.append(
                f"{len(senior_buys)} of those buys came from senior leadership "
                f"({', '.join(sorted({t.position for t in senior_buys}))})."
            )
        if senior_sells:
            reasons.append(
                f"{len(senior_sells)} sells came from senior leadership "
                f"({', '.join(sorted({t.position for t in senior_sells}))})."
            )
        if grants:
            reasons.append(
                f"Excluded {len(grants)} compensation grants from the score "
                f"(awards and vesting are pay, not conviction)."
            )

    notes = (
        f"Insider activity for {symbol} reads {signal} over the last "
        f"{lookback_days} days, from {len(transactions)} filings."
    )

    return InsiderRead(
        symbol=symbol,
        signal=signal,
        lookback_days=lookback_days,
        buy_count=len(buys),
        sell_count=len(sells),
        buy_value=buy_value,
        sell_value=sell_value,
        net_value=net_value,
        senior_buy_count=len(senior_buys),
        senior_sell_count=len(senior_sells),
        transactions=transactions,
        notes=notes,
        reasons=reasons,
    )
