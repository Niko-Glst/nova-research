"""Entry/exit strategy logic.

Turns price history into concrete, backtestable entry/exit signals. Produces
boolean Series aligned to the price index, consumable by
backtests/backtest_engine.py. This module never places a real order; it only
emits research signals.

Three strategies are provided, all computed from the same indicators the
technical analyst uses, so a backtest is testing the pipeline's own logic rather
than an unrelated rule:

- `moving_average_crossover`: long while the fast average is above the slow one.
  The classic trend-following rule, included mainly as a baseline to measure
  anything else against.
- `thesis_signals`: enters when the technical read turns constructive and the
  trend regime agrees, exits when either fails.
- `donchian_breakout`: Turtle-style channel breakout, with a volume filter so a
  new high on thin trading does not count as a breakout.

Every signal here is computed from data available *at that bar*, and the
backtest shifts them forward one bar before trading. Between them, that keeps
look-ahead out of the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.ta_analyst import (
    DONCHIAN_PERIOD,
    MACD_NOISE_FLOOR_PCT,
    RSI_OVERBOUGHT,
    RSI_PERIOD,
    SMA_FAST,
    SMA_SLOW,
    VOLUME_SURGE_RATIO,
    _donchian,
    _macd,
    _rsi,
    _volume_ratio,
)


@dataclass(frozen=True)
class StrategySignals:
    symbol: str
    entries: pd.Series  # boolean series aligned to price index
    exits: pd.Series  # boolean series aligned to price index
    rationale: str

    @property
    def entry_count(self) -> int:
        return int(self.entries.sum())

    @property
    def exit_count(self) -> int:
        return int(self.exits.sum())


def moving_average_crossover(
    price_history: pd.DataFrame,
    symbol: str = "",
    fast: int = SMA_FAST,
    slow: int = SMA_SLOW,
) -> StrategySignals:
    """Long while the fast moving average sits above the slow one.

    Entries fire on the bar the crossover happens rather than on every bar of
    the regime, so the backtest records one trade per regime rather than one
    per day.
    """
    close = price_history["Close"].astype(float)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    above = fast_ma > slow_ma
    # A crossover is the first bar of a new regime: True now, False before.
    entries = above & ~above.shift(1, fill_value=False)
    exits = ~above & above.shift(1, fill_value=False)

    # Bars without enough history to compute both averages cannot signal.
    valid = fast_ma.notna() & slow_ma.notna()
    entries &= valid
    exits &= valid

    return StrategySignals(
        symbol=symbol,
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        rationale=(
            f"Long while the {fast}-day average is above the {slow}-day average; "
            f"flat otherwise."
        ),
    )


def thesis_signals(price_history: pd.DataFrame, symbol: str = "") -> StrategySignals:
    """Enter when trend and momentum agree; exit when either turns.

    Entry requires all three: price above its fast average, the fast average
    above the slow one (an uptrend regime), and a positive MACD histogram beyond
    the noise floor. Exit when price loses the fast average, or momentum turns
    negative, or RSI reaches overbought.
    """
    close = price_history["Close"].astype(float)
    fast_ma = close.rolling(SMA_FAST).mean()
    slow_ma = close.rolling(SMA_SLOW).mean()
    macd_line, macd_signal_line = _macd(close)
    histogram = macd_line - macd_signal_line
    rsi = _rsi(close, RSI_PERIOD)

    # Scale the momentum floor to price, as the technical analyst does.
    floor = close.abs() * MACD_NOISE_FLOOR_PCT

    uptrend = (close > fast_ma) & (fast_ma > slow_ma)
    momentum_up = histogram > floor
    constructive = uptrend & momentum_up

    breakdown = (close < fast_ma) | (histogram < -floor) | (rsi >= RSI_OVERBOUGHT)

    entries = constructive & ~constructive.shift(1, fill_value=False)
    exits = breakdown & ~breakdown.shift(1, fill_value=False)

    valid = fast_ma.notna() & slow_ma.notna() & rsi.notna()
    entries &= valid
    exits &= valid

    return StrategySignals(
        symbol=symbol,
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        rationale=(
            f"Enter when price is above its {SMA_FAST}-day average within a "
            f"{SMA_FAST}/{SMA_SLOW} uptrend and MACD momentum is positive; exit on "
            f"a loss of the {SMA_FAST}-day average, negative momentum, or RSI "
            f">= {RSI_OVERBOUGHT:.0f}."
        ),
    )


def donchian_breakout(
    price_history: pd.DataFrame,
    symbol: str = "",
    entry_period: int = DONCHIAN_PERIOD,
    exit_period: int = 10,
    require_volume: bool = True,
) -> StrategySignals:
    """Classic Turtle-style breakout: buy new highs, exit on a shorter-window low.

    Entry and exit use different windows on purpose. A symmetric channel exits
    a position at the same level it would re-enter, which churns; a shorter exit
    window (10 days against 20) gets out of a failing breakout sooner while
    still letting a working one run.

    With `require_volume`, a breakout must also come on above-average volume.
    A new high on thin volume is a handful of trades finding no sellers, not a
    change in demand -- filtering those out is the main thing that separates a
    breakout system from a noise generator.
    """
    close = price_history["Close"].astype(float)
    high = price_history.get("High", close).astype(float)
    low = price_history.get("Low", close).astype(float)

    entry_upper, _ = _donchian(high, low, entry_period)
    _, exit_lower = _donchian(high, low, exit_period)

    breaking_out = close > entry_upper
    breaking_down = close < exit_lower

    if require_volume and "Volume" in price_history.columns:
        ratio = _volume_ratio(price_history["Volume"].astype(float))
        confirmed = ratio >= VOLUME_SURGE_RATIO
        breaking_out &= confirmed.fillna(False)

    # Fire on the first bar of a break, not on every bar above the channel.
    entries = breaking_out & ~breaking_out.shift(1, fill_value=False)
    exits = breaking_down & ~breaking_down.shift(1, fill_value=False)

    valid = entry_upper.notna() & exit_lower.notna()
    entries &= valid
    exits &= valid

    volume_note = (
        f" Breakouts require volume at {VOLUME_SURGE_RATIO:.1f}x its average."
        if require_volume
        else ""
    )
    return StrategySignals(
        symbol=symbol,
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        rationale=(
            f"Enter on a close above the {entry_period}-day Donchian high; exit "
            f"on a close below the {exit_period}-day low.{volume_note}"
        ),
    )


def generate_signals(
    symbol: str, thesis: dict, price_history: pd.DataFrame
) -> StrategySignals:
    """Generate entry/exit signals for a symbol given its thesis and price history.

    Kept for the interface the pipeline scaffold defines. `thesis` may carry a
    "strategy" key selecting between the rules above; it defaults to the
    thesis-driven one.
    """
    strategy = (thesis or {}).get("strategy", "thesis")
    if strategy == "crossover":
        return moving_average_crossover(price_history, symbol=symbol)
    if strategy == "breakout":
        return donchian_breakout(price_history, symbol=symbol)
    return thesis_signals(price_history, symbol=symbol)
