"""Point-in-time market data for the signal-weight backtest: OHLCV and earnings.

Two inputs, both cached locally under data/cache/ (gitignored) so a walk-forward
study over twenty symbols does not refetch on every run:

- **OHLCV** through agents.ta_analyst.fetch_price_history (rate-limited,
  dividend- and split-adjusted), indexed by local exchange date.
- **Earnings history** from yfinance: each announcement's timestamp, the
  consensus EPS estimate, the reported EPS and the surprise. The timestamp
  matters: a report at 16:00 New York time is after the close and can only be
  traded the next day, one at 08:00 is known before that day's close. Each
  surprise is stamped with the first trading day on whose close it was public
  (see `available_from`), which keeps look-ahead out of the backtest.

Fundamentals proper (P/E, margins, growth) are not here: yfinance only serves
the last five quarters, far too short to backtest. Earnings surprises are the
fundamental signal that does have a point-in-time history.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from agents.ta_analyst import fetch_price_history
from common.files import atomic_write, atomic_write_text
from common.rate_limit import get_limiter

ROOT_DIR = Path(__file__).resolve().parent.parent
OHLCV_CACHE = ROOT_DIR / "data" / "cache" / "ohlcv"
EARNINGS_CACHE = ROOT_DIR / "data" / "cache" / "earnings"

# Same rule as the correlation module's price cache: an evening fetch is fresh
# for a run the next morning; the following evening refetches.
CACHE_MAX_AGE_HOURS = 18

# Earnings change four times a year; a day-old copy is fine.
EARNINGS_MAX_AGE_HOURS = 24

# yfinance returns the most recent N announcements; Yahoo refuses more than
# 100 ("Yahoo caps limit at 100"), which reaches back to about 2002 for a
# quarterly reporter.
EARNINGS_LIMIT = 100

# US regular session close, New York time. An announcement at or after this
# time is first tradable at the next session's close.
MARKET_CLOSE_NY = 16

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

Fetcher = Callable[[str, str], pd.DataFrame]


def _stem(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)


def _fresh(meta_path: Path, max_age_hours: float, now: datetime) -> bool:
    if not meta_path.exists():
        return False
    try:
        fetched = datetime.fromisoformat(json.loads(meta_path.read_text())["fetched_at"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return False
    return now - fetched <= timedelta(hours=max_age_hours)


def _write_meta(meta_path: Path, now: datetime, rows: int) -> None:
    atomic_write_text(meta_path, json.dumps({"fetched_at": now.isoformat(), "rows": rows}))


def load_ohlcv(
    symbol: str,
    refresh: bool = False,
    cache_dir: Path = OHLCV_CACHE,
    fetcher: Fetcher = fetch_price_history,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Daily OHLCV indexed by local exchange date, from the cache when fresh."""
    symbol = symbol.strip().upper()
    now = now or datetime.now(timezone.utc)
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / f"{_stem(symbol)}.csv"
    meta_path = cache_dir / f"{_stem(symbol)}.meta.json"

    if not refresh and csv_path.exists() and _fresh(meta_path, CACHE_MAX_AGE_HOURS, now):
        return pd.read_csv(csv_path, index_col="date", parse_dates=["date"])

    try:
        history = fetcher(symbol, "max")
    except Exception:
        if csv_path.exists():  # stale prices are still real prices
            return pd.read_csv(csv_path, index_col="date", parse_dates=["date"])
        raise

    frame = history[[c for c in OHLCV_COLUMNS if c in history.columns]].astype(float).copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the exchange's own calendar date
    frame.index = index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame[frame["Close"] > 0].dropna(subset=["Close"])
    frame.index.name = "date"

    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(csv_path, frame.to_csv)  # never a half-written history read as fresh
    _write_meta(meta_path, now, len(frame))
    return frame


def fetch_earnings(symbol: str) -> pd.DataFrame:
    """Raw earnings history from yfinance, rate-limited."""
    get_limiter("yfinance").wait()
    frame = yf.Ticker(symbol.strip().upper()).get_earnings_dates(limit=EARNINGS_LIMIT)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["announced", "eps_estimate", "eps_reported", "surprise_pct"])
    out = pd.DataFrame(
        {
            "announced": pd.DatetimeIndex(frame.index),
            "eps_estimate": pd.to_numeric(frame.get("EPS Estimate"), errors="coerce").to_numpy(),
            "eps_reported": pd.to_numeric(frame.get("Reported EPS"), errors="coerce").to_numpy(),
            "surprise_pct": pd.to_numeric(frame.get("Surprise(%)"), errors="coerce").to_numpy(),
        }
    )
    # Future announcement dates carry no reported EPS yet; they are not data.
    return out.dropna(subset=["eps_reported"]).reset_index(drop=True)


def load_earnings(
    symbol: str,
    refresh: bool = False,
    cache_dir: Path = EARNINGS_CACHE,
    fetcher: Callable[[str], pd.DataFrame] = fetch_earnings,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Earnings history with New York announcement timestamps, cached."""
    symbol = symbol.strip().upper()
    now = now or datetime.now(timezone.utc)
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / f"{_stem(symbol)}.csv"
    meta_path = cache_dir / f"{_stem(symbol)}.meta.json"

    if not refresh and csv_path.exists() and _fresh(meta_path, EARNINGS_MAX_AGE_HOURS, now):
        frame = pd.read_csv(csv_path)
    else:
        try:
            frame = fetcher(symbol)
        except Exception:
            if not csv_path.exists():
                raise
            frame = pd.read_csv(csv_path)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(csv_path, lambda p: frame.to_csv(p, index=False))
            _write_meta(meta_path, now, len(frame))

    announced = pd.to_datetime(frame["announced"], utc=True).dt.tz_convert("America/New_York")
    frame = frame.assign(announced=announced)
    return frame.sort_values("announced").reset_index(drop=True)


def available_from(announced: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day on whose close each announcement was public.

    Before 16:00 New York time (pre-market or intraday): that day's close.
    At or after 16:00: the next trading day's close. A date that is not a
    trading day rolls forward to the next one.
    """
    local = announced.dt.tz_localize(None) if announced.dt.tz is not None else announced
    day = local.dt.normalize()
    after_close = local.dt.hour >= MARKET_CLOSE_NY
    candidate = day + pd.to_timedelta(after_close.astype(int), unit="D")
    positions = trading_days.searchsorted(candidate.to_numpy(), side="left")
    result = pd.Series(pd.NaT, index=announced.index, dtype="datetime64[ns]")
    valid = positions < len(trading_days)
    result[valid] = trading_days[positions[valid]]
    return result
