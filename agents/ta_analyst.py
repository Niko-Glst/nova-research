"""Technical analysis on price data.

Fetches OHLCV data via yfinance and produces a structured technical read for a
given symbol. Indicators are computed directly in pandas rather than via
pandas-ta: that package is unmaintained and breaks on numpy 2.x, and the
handful of indicators this pipeline needs are a few lines each.

Research output only — nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from common.rate_limit import get_limiter

# Indicator windows. Conventional defaults; changing them changes every
# downstream signal, so they live here rather than being scattered inline.
SMA_FAST = 50
SMA_SLOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# Donchian channel lookback, in trading days. 20 is the classic Turtle-system
# window: long enough that a break means something, short enough to still be
# tradable. The breakout is measured against the prior N bars, excluding today,
# so "price made a new 20-day high" cannot be true by construction.
DONCHIAN_PERIOD = 20

# Volume confirmation windows. Today's volume is compared against its own
# recent average; a move on heavy volume carries conviction that the same move
# on thin volume does not.
VOLUME_PERIOD = 20
VOLUME_SURGE_RATIO = 1.5   # >= this much of average is a conviction move
VOLUME_WEAK_RATIO = 0.6    # <= this much means the move lacks participation

# Vote weights for the classifier.
#
# Trend evidence outranks everything: where price sits relative to its own
# averages is the most durable read available from price alone.
#
# Volume and Donchian rank above the oscillators. Volume shows conviction --
# whether anyone actually acted on the move -- which a price-only indicator
# cannot see. A Donchian break is an unambiguous, non-lagging statement that
# price has left its recent range, where MACD only reports that two averages
# crossed some time ago.
#
# MACD is deliberately the smallest vote. It is a lagging function of two
# exponential averages, it flips on small moves, and in this project's own
# signal study it was the component most prone to firing on noise. It is kept
# because a momentum cross adds something at the margin, not because it earns
# equal standing with the trend.
WEIGHT_PRICE_VS_SMA = 1.5
WEIGHT_SMA_CROSS = 1.5
WEIGHT_DONCHIAN = 1.25
WEIGHT_VOLUME = 1.0
WEIGHT_RSI = 0.75
WEIGHT_MACD = 0.5

# A MACD histogram this small relative to price is noise, not momentum. Without
# this floor, a histogram of +0.01 on a $16 stock casts a full bullish vote.
MACD_NOISE_FLOOR_PCT = 0.001  # 0.1% of price


@dataclass(frozen=True)
class TechnicalRead:
    symbol: str
    indicators: dict[str, float]
    signal: str  # "bullish" / "bearish" / "neutral"
    notes: str
    reasons: list[str] = field(default_factory=list)


def fetch_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV history for a symbol via yfinance, rate-limited.

    Returns a DataFrame indexed by date with Open/High/Low/Close/Volume columns.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    get_limiter("yfinance").wait()
    ticker = yf.Ticker(symbol.strip().upper())
    history = ticker.history(period=period, auto_adjust=True)

    if history is None or history.empty:
        raise ValueError(
            f"No price history returned for {symbol!r} (period={period!r}). "
            f"Check that the ticker exists and is spelled correctly."
        )

    # yfinance can hand back a MultiIndex when multiple tickers are requested;
    # flatten defensively so downstream code always sees plain column names.
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    return history.dropna(subset=["Close"])


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, smoothed with an exponentially weighted mean."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    # A zero average loss means an unbroken run of gains: RSI is 100 by definition.
    return rsi.fillna(100.0).where(avg_gain.notna(), other=float("nan"))


def _donchian(
    high: pd.Series, low: pd.Series, period: int = DONCHIAN_PERIOD
) -> tuple[pd.Series, pd.Series]:
    """Upper and lower Donchian channel over the PRIOR `period` bars.

    The window is shifted by one bar so today's own high and low are excluded.
    Without that shift the upper channel always equals today's high on a new
    high, and "price broke out" would be trivially true whenever it happened to
    be the highest bar -- the channel has to describe the range price is
    breaking out *of*.
    """
    upper = high.rolling(period).max().shift(1)
    lower = low.rolling(period).min().shift(1)
    return upper, lower


def _volume_ratio(volume: pd.Series, period: int = VOLUME_PERIOD) -> pd.Series:
    """Volume as a multiple of its own trailing average, excluding today.

    Today is excluded from the average for the same reason as the Donchian
    window: including it dampens exactly the spike being measured.
    """
    average = volume.rolling(period).mean().shift(1)
    return volume / average.replace(0.0, pd.NA)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the MACD line and its signal line."""
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line, signal_line


def compute_indicators(price_history: pd.DataFrame) -> dict[str, float]:
    """Compute technical indicators for the given price history.

    Returns the latest value of each indicator. Values that need more history
    than is available are omitted rather than returned as NaN.
    """
    if price_history.empty:
        raise ValueError("price_history is empty — nothing to compute")

    close = price_history["Close"].astype(float)
    macd_line, macd_signal = _macd(close)

    # High/low drive the Donchian channel; fall back to close when a provider
    # omits them, which turns the channel into a close-based range rather than
    # dropping the indicator entirely.
    high = price_history.get("High", close).astype(float)
    low = price_history.get("Low", close).astype(float)
    donchian_upper, donchian_lower = _donchian(high, low)

    series: dict[str, pd.Series] = {
        "close": close,
        f"sma_{SMA_FAST}": close.rolling(SMA_FAST).mean(),
        f"sma_{SMA_SLOW}": close.rolling(SMA_SLOW).mean(),
        f"rsi_{RSI_PERIOD}": _rsi(close),
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_line - macd_signal,
        f"donchian_upper_{DONCHIAN_PERIOD}": donchian_upper,
        f"donchian_lower_{DONCHIAN_PERIOD}": donchian_lower,
    }

    if "Volume" in price_history.columns:
        volume = price_history["Volume"].astype(float)
        series["volume"] = volume
        series[f"volume_avg_{VOLUME_PERIOD}"] = volume.rolling(VOLUME_PERIOD).mean()
        series["volume_ratio"] = _volume_ratio(volume)

    indicators: dict[str, float] = {}
    for name, values in series.items():
        latest = values.iloc[-1]
        if pd.notna(latest):
            indicators[name] = float(latest)

    # Trailing return over the window we actually have, as a rough momentum read.
    if len(close) > 1:
        indicators["return_pct"] = float(
            (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
        )

    return indicators


def _classify(indicators: dict[str, float]) -> tuple[str, list[str]]:
    """Turn indicator values into a stance plus the reasons behind it.

    Each check casts a weighted vote (see the WEIGHT_* constants); the sign of
    the total decides the stance. A total within `dead_zone` of zero reads
    neutral, so a stock with genuinely mixed evidence is not forced into a
    direction by one marginal vote.
    """
    reasons: list[str] = []
    score = 0.0
    cast = 0.0

    close = indicators.get("close")
    sma_fast = indicators.get(f"sma_{SMA_FAST}")
    sma_slow = indicators.get(f"sma_{SMA_SLOW}")

    if close is not None and sma_fast is not None:
        cast += WEIGHT_PRICE_VS_SMA
        if close > sma_fast:
            score += WEIGHT_PRICE_VS_SMA
            reasons.append(f"Price is above its {SMA_FAST}-day average (short-term uptrend).")
        else:
            score -= WEIGHT_PRICE_VS_SMA
            reasons.append(f"Price is below its {SMA_FAST}-day average (short-term downtrend).")

    if sma_fast is not None and sma_slow is not None:
        cast += WEIGHT_SMA_CROSS
        if sma_fast > sma_slow:
            score += WEIGHT_SMA_CROSS
            reasons.append(
                f"{SMA_FAST}-day average is above the {SMA_SLOW}-day (golden-cross regime)."
            )
        else:
            score -= WEIGHT_SMA_CROSS
            reasons.append(
                f"{SMA_FAST}-day average is below the {SMA_SLOW}-day (death-cross regime)."
            )

    # Donchian breakout: has price left the range it has held for 20 days?
    upper = indicators.get(f"donchian_upper_{DONCHIAN_PERIOD}")
    lower = indicators.get(f"donchian_lower_{DONCHIAN_PERIOD}")
    if close is not None and upper is not None and lower is not None:
        cast += WEIGHT_DONCHIAN
        if close > upper:
            score += WEIGHT_DONCHIAN
            reasons.append(
                f"Price broke above its {DONCHIAN_PERIOD}-day high "
                f"({upper:.2f}): a breakout from the recent range."
            )
        elif close < lower:
            score -= WEIGHT_DONCHIAN
            reasons.append(
                f"Price broke below its {DONCHIAN_PERIOD}-day low "
                f"({lower:.2f}): a breakdown from the recent range."
            )
        else:
            position = (
                (close - lower) / (upper - lower) if upper > lower else 0.5
            )
            reasons.append(
                f"Price sits {position:.0%} of the way up its "
                f"{DONCHIAN_PERIOD}-day range, with no breakout."
            )

    # Volume confirmation: conviction behind the move, not the move itself.
    # This votes with the direction of the day rather than on its own, because
    # heavy volume is only bullish if price is rising on it.
    volume_ratio = indicators.get("volume_ratio")
    if volume_ratio is not None and close is not None and sma_fast is not None:
        cast += WEIGHT_VOLUME
        direction = 1.0 if close > sma_fast else -1.0
        if volume_ratio >= VOLUME_SURGE_RATIO:
            score += WEIGHT_VOLUME * direction
            reasons.append(
                f"Volume is {volume_ratio:.1f}x its {VOLUME_PERIOD}-day average: "
                f"real participation behind "
                f"{'the advance' if direction > 0 else 'the decline'}."
            )
        elif volume_ratio <= VOLUME_WEAK_RATIO:
            # Thin volume argues against whatever the trend is claiming, so it
            # votes against the prevailing direction rather than with it.
            score -= WEIGHT_VOLUME * direction * 0.5
            reasons.append(
                f"Volume is only {volume_ratio:.1f}x its {VOLUME_PERIOD}-day "
                f"average: the move lacks conviction."
            )
        else:
            reasons.append(
                f"Volume is {volume_ratio:.1f}x its {VOLUME_PERIOD}-day average "
                f"(unremarkable)."
            )

    rsi = indicators.get(f"rsi_{RSI_PERIOD}")
    if rsi is not None:
        cast += WEIGHT_RSI
        if rsi >= RSI_OVERBOUGHT:
            score -= WEIGHT_RSI
            reasons.append(f"RSI {rsi:.1f} is overbought (>= {RSI_OVERBOUGHT:.0f}).")
        elif rsi <= RSI_OVERSOLD:
            score += WEIGHT_RSI
            reasons.append(f"RSI {rsi:.1f} is oversold (<= {RSI_OVERSOLD:.0f}).")
        else:
            reasons.append(f"RSI {rsi:.1f} is in neutral territory.")

    histogram = indicators.get("macd_histogram")
    if histogram is not None:
        cast += WEIGHT_MACD
        # Scale the noise floor to the share price: 0.01 means something on a
        # $2 stock and nothing on a $500 one.
        floor = abs(close) * MACD_NOISE_FLOOR_PCT if close else 0.0
        if abs(histogram) <= floor:
            reasons.append(
                f"MACD histogram ({histogram:+.3f}) is within noise of its signal line."
            )
        elif histogram > 0:
            score += WEIGHT_MACD
            reasons.append("MACD is above its signal line (positive momentum).")
        else:
            score -= WEIGHT_MACD
            reasons.append("MACD is below its signal line (negative momentum).")

    # Require the net vote to clear a fifth of the weight actually cast, so a
    # single marginal signal cannot decide an otherwise balanced picture.
    dead_zone = cast * 0.2

    if score > dead_zone:
        return "bullish", reasons
    if score < -dead_zone:
        return "bearish", reasons
    return "neutral", reasons


def analyze(symbol: str, period: str = "1y") -> TechnicalRead:
    """Run the full TA pipeline for a symbol."""
    symbol = symbol.strip().upper()
    price_history = fetch_price_history(symbol, period=period)
    indicators = compute_indicators(price_history)
    signal, reasons = _classify(indicators)

    days = len(price_history)
    notes = (
        f"{signal.capitalize()} technical read for {symbol} over {days} trading days "
        f"({period})."
    )
    if days < SMA_SLOW:
        notes += (
            f" Note: only {days} days of history, so the {SMA_SLOW}-day average "
            f"could not be computed."
        )

    return TechnicalRead(
        symbol=symbol,
        indicators=indicators,
        signal=signal,
        notes=notes,
        reasons=reasons,
    )
