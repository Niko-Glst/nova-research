"""The signal register: every input the weighted strategy can use, with its own weight.

A strategy here is a weighted sum of signals. Each signal is a daily series in
[-1, +1] computed only from data available at that day's close: +1 is the
strongest buy reading, -1 the strongest sell, 0 no view. The backtest acts on
it the next day.

**Adding a parameter later** is one function and one `register` call::

    def _my_signal(inputs: SignalInputs) -> pd.Series:
        close = inputs.ohlcv["Close"]
        return (close.pct_change(5) / 0.05).clip(-1, 1)

    register(Signal("my_signal", "technical", "Five-day return, scaled", _my_signal))

The optimiser then finds its weight alongside the others; nothing else
changes. Signals with no point-in-time history (sentiment and insider data
today) are registered with `point_in_time=False`. They are listed, with the
reason, but kept out of the backtest until their history exists:
agents/signal_snapshots.py now records them on every research run, so they
become testable as that record grows.

Every scale below is a fixed constant chosen in advance, never fitted to the
data: a scale estimated from the full history would leak future information
into every past signal value.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.ta_analyst import (
    DONCHIAN_PERIOD,
    RSI_PERIOD,
    SMA_FAST,
    SMA_SLOW,
    _donchian,
    _macd,
    _rsi,
    _volume_ratio,
)
from backtests.market_data import available_from

GROUPS = ("technical", "fundamental", "sentiment", "insider", "custom")

# Scales: the move that maps to a full +-1 reading.
TREND_SCALE = 0.10  # 50-day average 10% above the 200-day is a full uptrend
PRICE_VS_SMA_SCALE = 0.05  # price 5% above its 50-day average
MOMENTUM_SCALE = 0.30  # 30% over the 12 months skipping the last one
MACD_SCALE = 0.005  # histogram worth 0.5% of price
SURPRISE_SCALE = 10.0  # a 10% earnings beat is a full reading

# Momentum skips the most recent month: one-month returns tend to reverse
# (Jegadeesh & Titman 1993), which would cancel the 12-month effect.
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21

# Post-earnings drift fades over roughly a quarter (Bernard & Thomas 1989).
# The surprise signal decays with this half-life until the next report.
SURPRISE_HALF_LIFE_DAYS = 30

# Volume signal: the move it confirms is the five-day return.
VOLUME_MOVE_DAYS = 5


@dataclass(frozen=True)
class SignalInputs:
    """Everything a signal may read. Values must be used only up to each date."""

    symbol: str
    ohlcv: pd.DataFrame
    earnings: pd.DataFrame | None = None


@dataclass(frozen=True)
class Signal:
    name: str
    group: str
    description: str
    compute: Callable[[SignalInputs], pd.Series] | None
    point_in_time: bool = True  # False: no history to backtest yet
    reason_unavailable: str = ""

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")
        if self.point_in_time and self.compute is None:
            raise ValueError(f"Signal {self.name!r} is point-in-time but has no compute function.")


REGISTRY: dict[str, Signal] = {}


def register(signal: Signal) -> Signal:
    if signal.name in REGISTRY:
        raise ValueError(f"A signal named {signal.name!r} is already registered.")
    REGISTRY[signal.name] = signal
    return signal


def backtestable() -> list[Signal]:
    return [s for s in REGISTRY.values() if s.point_in_time]


def unavailable() -> list[Signal]:
    return [s for s in REGISTRY.values() if not s.point_in_time]


def compute_all(inputs: SignalInputs, names: list[str] | None = None) -> pd.DataFrame:
    """Every requested backtestable signal for one symbol, one column each."""
    chosen = [REGISTRY[n] for n in names] if names else backtestable()
    columns = {}
    for signal in chosen:
        if not signal.point_in_time:
            raise ValueError(f"{signal.name} has no point-in-time history: "
                             f"{signal.reason_unavailable}")
        series = signal.compute(inputs).reindex(inputs.ohlcv.index)
        columns[signal.name] = series.clip(-1.0, 1.0)
    return pd.DataFrame(columns, index=inputs.ohlcv.index)


# --------------------------------------------------------------------------
# Technical signals
# --------------------------------------------------------------------------


def _close(inputs: SignalInputs) -> pd.Series:
    return inputs.ohlcv["Close"].astype(float)


def _trend(inputs: SignalInputs) -> pd.Series:
    close = _close(inputs)
    fast, slow = close.rolling(SMA_FAST).mean(), close.rolling(SMA_SLOW).mean()
    return (fast / slow - 1.0) / TREND_SCALE


def _price_vs_sma(inputs: SignalInputs) -> pd.Series:
    close = _close(inputs)
    return (close / close.rolling(SMA_FAST).mean() - 1.0) / PRICE_VS_SMA_SCALE


def _momentum(inputs: SignalInputs) -> pd.Series:
    close = _close(inputs)
    return (close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1.0) / MOMENTUM_SCALE


def _macd_signal(inputs: SignalInputs) -> pd.Series:
    close = _close(inputs)
    line, signal_line = _macd(close)
    return (line - signal_line) / (close * MACD_SCALE)


def _rsi_reversion(inputs: SignalInputs) -> pd.Series:
    """Contrarian: oversold (RSI 30) reads +0.4, overbought (RSI 70) reads -0.4."""
    return (50.0 - _rsi(_close(inputs), RSI_PERIOD)) / 50.0


def _channel_position(inputs: SignalInputs) -> pd.Series:
    """Where the close sits in the prior 20-day high-low range: -1 bottom, +1 top."""
    frame = inputs.ohlcv
    close = _close(inputs)
    upper, lower = _donchian(frame.get("High", close), frame.get("Low", close), DONCHIAN_PERIOD)
    width = (upper - lower).replace(0.0, np.nan)
    return 2.0 * (close - lower) / width - 1.0


def _volume_confirmation(inputs: SignalInputs) -> pd.Series:
    """Direction of the last week's move, scaled by how unusual today's volume is."""
    frame = inputs.ohlcv
    if "Volume" not in frame:
        return pd.Series(np.nan, index=frame.index)
    close = _close(inputs)
    ratio = pd.to_numeric(_volume_ratio(frame["Volume"].astype(float)), errors="coerce")
    strength = (np.log2(ratio.clip(lower=1.0))).clip(upper=1.0)  # 2x average volume = full
    return np.sign(close.pct_change(VOLUME_MOVE_DAYS)) * strength


# --------------------------------------------------------------------------
# Fundamental signal: earnings surprise with post-announcement drift
# --------------------------------------------------------------------------


def _earnings_surprise(inputs: SignalInputs) -> pd.Series:
    """The latest surprise, from the close it became public, decaying until the next.

    Before the first known report the value is NaN (no information), not zero.
    """
    index = inputs.ohlcv.index
    out = pd.Series(np.nan, index=index)
    earnings = inputs.earnings
    if earnings is None or earnings.empty:
        return out
    frame = earnings.dropna(subset=["surprise_pct"]).copy()
    if frame.empty:
        return out
    frame["from"] = available_from(frame["announced"], index)
    frame = frame.dropna(subset=["from"]).sort_values("from").drop_duplicates("from", keep="last")

    position = np.arange(len(index))
    starts = index.get_indexer(pd.DatetimeIndex(frame["from"]))
    level = pd.Series(np.nan, index=index)
    start_pos = pd.Series(np.nan, index=index)
    level.iloc[starts] = (frame["surprise_pct"].to_numpy() / SURPRISE_SCALE)
    start_pos.iloc[starts] = starts
    level, start_pos = level.ffill(), start_pos.ffill()
    age = position - start_pos.to_numpy()
    decay = 0.5 ** (age / SURPRISE_HALF_LIFE_DAYS)
    return pd.Series(level.to_numpy() * decay, index=index)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


register(Signal("trend", "technical",
                f"{SMA_FAST}-day average against the {SMA_SLOW}-day: the crossover rule's "
                f"input, as a strength rather than an on/off switch", _trend))
register(Signal("price_vs_sma50", "technical",
                f"Close against its {SMA_FAST}-day average", _price_vs_sma))
register(Signal("momentum_12_1", "technical",
                "Return over the past 12 months, skipping the latest month", _momentum))
register(Signal("macd", "technical", "MACD histogram relative to price", _macd_signal))
register(Signal("rsi_reversion", "technical",
                "Contrarian RSI: positive when oversold, negative when overbought",
                _rsi_reversion))
register(Signal("channel_position", "technical",
                f"Position of the close in the prior {DONCHIAN_PERIOD}-day high-low channel",
                _channel_position))
register(Signal("volume_confirmation", "technical",
                "Direction of the past week's move, weighted by unusual volume",
                _volume_confirmation))
register(Signal("earnings_surprise", "fundamental",
                f"Latest EPS surprise against consensus, decaying with a "
                f"{SURPRISE_HALF_LIFE_DAYS}-day half-life (post-earnings drift)",
                _earnings_surprise))

register(Signal("fundamental_valuation", "fundamental",
                "P/E, margins and growth against niche peers (agents/fa_analyst.py)", None,
                point_in_time=False,
                reason_unavailable="yfinance serves only the last five quarters of statements"))
register(Signal("news_sentiment", "sentiment",
                "Recency-weighted news sentiment (agents/sentiment_analyst.py)", None,
                point_in_time=False,
                reason_unavailable="news history reaches back about a year; being recorded "
                                   "from now on"))
register(Signal("reddit_sentiment", "sentiment",
                "Upvote-weighted Reddit sentiment", None, point_in_time=False,
                reason_unavailable="Reddit search covers the past month; being recorded "
                                   "from now on"))
register(Signal("insider_flow", "insider",
                "Open-market insider buying minus selling (agents/insider_analyst.py)", None,
                point_in_time=False,
                reason_unavailable="yfinance serves about two years of filings; being "
                                   "recorded from now on"))
