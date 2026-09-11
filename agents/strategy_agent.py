"""Entry/exit strategy logic.

Turns price history into concrete, backtestable entry/exit signals. Produces
boolean Series aligned to the price index, consumable by
backtests/backtest_engine.py. This module never places a real order; it only
emits research signals.

Two strategies are provided, both computed from the same indicators the
technical analyst uses, so a backtest is testing the pipeline's own logic rather
than an unrelated rule:

- `moving_average_crossover`: long while the fast average is above the slow one.
  The classic trend-following rule, included mainly as a baseline to measure
  anything else against.
- `thesis_signals`: enters when the technical read turns constructive and the
  trend regime agrees, exits when either fails.

Every signal here is computed from data available *at that bar*, and the
backtest shifts them forward one bar before trading. Between them, that keeps
look-ahead out of the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.ta_analyst import (
    MACD_NOISE_FLOOR_PCT,
    RSI_OVERBOUGHT,
    RSI_PERIOD,
    SMA_FAST,
    SMA_SLOW,
    _macd,
    _rsi,
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
    return thesis_signals(price_history, symbol=symbol)
