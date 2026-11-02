"""Position-trading rule: a pullback in an uptrend, taken only when every condition holds.

Built on ThinkMarkets' guide "What is position trading" (owner, 2026-10-28):
trends lasting weeks to months, moving averages 50/100/200 for the trend and as
dynamic support, support and resistance from a higher (weekly) timeframe,
Fibonacci retracements for pullback levels, and a momentum trigger (MACD over
its signal line, RSI back above 50). Stops sit at swings, and are raised to
"secure profits while allowing further advances".

**Entry** (all must hold; config/scoring.yaml `position_strategy`):

1. Uptrend: close above the 200-day average and the 50-day above the 200-day.
2. Pullback: the close sits in the 50-61.8% retracement of the last up-leg
   (last confirmed weekly swing low to the confirmed weekly swing high after it).
3. Support: a weekly swing low, or the 100- or 200-day average, lies within 2%
   below the close.
4b. Bollinger (owner's addition, 2026-10-29; not in the guide): within the setup
   window a close at or below the lower 20/2 band, and the close back inside the
   bands when the trigger fires -- the dip ran out of selling.
4. Trigger: within 10 sessions of conditions 1-3 holding, MACD crosses above its
   signal line or RSI crosses back above 50, with the uptrend still intact and
   the pullback not broken: the close may have bounced above the zone, but not
   fallen below the 61.8% level.

**Exit**: a close below the stop, or below the 200-day average. The stop starts
at the up-leg's swing low and is raised to each higher confirmed weekly swing low.

**Point in time**: a weekly swing counts only once the weeks that confirm it have
closed, so no turning point is used before it was visible. Signals come from a
close and are traded at the next bar (backtest_engine shifts them).

Fundamentals, central to position trading in the guide, have no point-in-time
history here and are not part of the rule. Research output only -- nothing here
places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.ta_analyst import _macd, _rsi
from agents.strategy_agent import StrategySignals
from config.settings import load_scoring_config

_CFG = load_scoring_config()["position_strategy"]
MA_FAST, MA_MID, MA_SLOW = int(_CFG["ma_fast"]), int(_CFG["ma_mid"]), int(_CFG["ma_slow"])
SWING_WEEKS = int(_CFG["swing_weeks"])
FIB_LOW, FIB_HIGH = (float(v) for v in _CFG["fib_zone"])
SUPPORT_TOLERANCE = float(_CFG["support_tolerance"])
SETUP_WINDOW = int(_CFG["setup_window"])
RSI_TRIGGER = float(_CFG["rsi_trigger"])
BOLLINGER_DAYS = int(_CFG["bollinger_days"])
BOLLINGER_SD = float(_CFG["bollinger_sd"])
FIB_EXTENSIONS = tuple(float(v) for v in _CFG["fib_extensions"])
FEES_BPS = float(_CFG["fees_bps"])

CONDITIONS = ("uptrend", "fib_zone", "support", "bollinger", "trigger")
CONDITION_TEXT = {
    "uptrend": f"Uptrend: close above the {MA_SLOW}-day, {MA_FAST}-day above the {MA_SLOW}-day",
    "fib_zone": f"Pullback in the {FIB_LOW:.0%}-{FIB_HIGH:.1%} retracement of the last up-leg",
    "support": f"Support within {SUPPORT_TOLERANCE:.0%} below (weekly swing low or "
               f"{MA_MID}/{MA_SLOW}-day)",
    "bollinger": f"Bollinger {BOLLINGER_DAYS}/{BOLLINGER_SD:g}: a close at or below the lower "
                 f"band in the last {SETUP_WINDOW} sessions, back inside now",
    "trigger": f"Momentum turns: MACD over its signal line or RSI back above "
               f"{RSI_TRIGGER:.0f}",
}


# --------------------------------------------------------------------------
# Weekly swings, point in time
# --------------------------------------------------------------------------


def weekly_swings(ohlc: pd.DataFrame, weeks: int = SWING_WEEKS) -> pd.DataFrame:
    """Confirmed weekly swing highs and lows, each with the day it became known.

    A week is a swing low when its low is the lowest of the `weeks` weeks either
    side (a swing high likewise). It is known only when the last of those later
    weeks has closed: `known_on` is that week's last trading day.
    """
    days = pd.Series(ohlc.index, index=ohlc.index)
    weekly = pd.DataFrame({
        "high": ohlc["High"].resample("W-FRI").max(),
        "low": ohlc["Low"].resample("W-FRI").min(),
        "last_day": days.resample("W-FRI").max(),
    }).dropna()
    rows = []
    span = 2 * weeks + 1
    for i in range(weeks, len(weekly) - weeks):
        window = weekly.iloc[i - weeks:i + weeks + 1]
        known_on = weekly["last_day"].iloc[i + weeks]
        week_day = weekly["last_day"].iloc[i]
        low, high = weekly["low"].iloc[i], weekly["high"].iloc[i]
        if len(window) == span and low == window["low"].min() \
                and (window["low"] == low).sum() == 1:
            rows.append({"kind": "low", "price": float(low), "week": week_day,
                         "known_on": known_on})
        if len(window) == span and high == window["high"].max() \
                and (window["high"] == high).sum() == 1:
            rows.append({"kind": "high", "price": float(high), "week": week_day,
                         "known_on": known_on})
    return pd.DataFrame(rows, columns=["kind", "price", "week", "known_on"])


@dataclass(frozen=True)
class Leg:
    low: float
    high: float
    low_week: pd.Timestamp
    high_week: pd.Timestamp

    def retracement(self, price: float) -> float:
        """0 at the leg's high, 1 at its low."""
        return (self.high - price) / (self.high - self.low)

    def level(self, fraction: float) -> float:
        return self.high - fraction * (self.high - self.low)

    def extension(self, multiple: float) -> float:
        return self.low + multiple * (self.high - self.low)


def leg_as_of(swings: pd.DataFrame, day: pd.Timestamp) -> Leg | None:
    """The last up-leg known on `day`: the latest swing high and the swing low before it."""
    known = swings[swings["known_on"] <= day]
    highs = known[known["kind"] == "high"]
    if highs.empty:
        return None
    top = highs.sort_values("week").iloc[-1]
    lows = known[(known["kind"] == "low") & (known["week"] < top["week"])]
    if lows.empty:
        return None
    bottom = lows.sort_values("week").iloc[-1]
    if not top["price"] > bottom["price"]:
        return None
    return Leg(float(bottom["price"]), float(top["price"]), bottom["week"], top["week"])


@dataclass(frozen=True)
class SwingState:
    """What was known about weekly swings on each day.

    The known set only grows when a swing is confirmed, a few dozen times in
    five years, so the leg and the sorted swing lows are built once per
    confirmation and each day looks up the latest one. (Recomputing them for
    every day took about 3 seconds a stock.)
    """
    event_days: np.ndarray  # confirmation days, ascending (datetime64)
    legs: tuple  # Leg | None in force from each event on
    lows: tuple  # sorted confirmed swing-low prices known from each event on
    highs: tuple  # sorted confirmed swing-high prices known from each event on

    def at(self, day) -> int:
        """Index of the latest event on or before `day`; -1 before the first."""
        return int(np.searchsorted(self.event_days, np.datetime64(day), side="right")) - 1

    def leg(self, day) -> Leg | None:
        i = self.at(day)
        return self.legs[i] if i >= 0 else None

    def lows_known(self, day) -> np.ndarray:
        i = self.at(day)
        return self.lows[i] if i >= 0 else np.empty(0)

    def highs_known(self, day) -> np.ndarray:
        i = self.at(day)
        return self.highs[i] if i >= 0 else np.empty(0)


def swing_state(swings: pd.DataFrame) -> SwingState:
    if swings.empty:
        return SwingState(np.array([], dtype="datetime64[ns]"), (), (), ())
    days = np.sort(swings["known_on"].unique())
    legs, lows, highs = [], [], []
    for day in days:
        known = swings[swings["known_on"] <= day]
        legs.append(leg_as_of(swings, pd.Timestamp(day)))
        lows.append(np.sort(known.loc[known["kind"] == "low", "price"].to_numpy(dtype=float)))
        highs.append(np.sort(known.loc[known["kind"] == "high", "price"].to_numpy(dtype=float)))
    return SwingState(np.asarray(days, dtype="datetime64[ns]"), tuple(legs), tuple(lows),
                      tuple(highs))


def _highest_at_or_below(values: np.ndarray, price: float) -> float | None:
    k = int(np.searchsorted(values, price, side="right"))
    return float(values[k - 1]) if k else None


# --------------------------------------------------------------------------
# Conditions per day and the rule
# --------------------------------------------------------------------------


def conditions(ohlc: pd.DataFrame, state: SwingState | None = None) -> pd.DataFrame:
    """Each condition per day, from data known on that day."""
    close = ohlc["Close"].astype(float)
    ma = {n: close.rolling(n).mean() for n in (MA_FAST, MA_MID, MA_SLOW)}
    state = state or swing_state(weekly_swings(ohlc))
    uptrend = (close > ma[MA_SLOW]) & (ma[MA_FAST] > ma[MA_SLOW])

    prices = close.to_numpy()
    mid_ma, slow_ma = ma[MA_MID].to_numpy(), ma[MA_SLOW].to_numpy()
    n = len(prices)
    retracement, leg_low = np.full(n, np.nan), np.full(n, np.nan)
    support = np.zeros(n, dtype=bool)
    for i, day in enumerate(close.index):
        price = prices[i]
        leg = state.leg(day)
        if leg is not None:
            retracement[i] = leg.retracement(price)
            leg_low[i] = leg.low
        levels = [v for v in (_highest_at_or_below(state.lows_known(day), price),
                              mid_ma[i], slow_ma[i])
                  if v is not None and np.isfinite(v) and v <= price]
        if levels:
            support[i] = (price - max(levels)) / price <= SUPPORT_TOLERANCE
    retracement = pd.Series(retracement, index=close.index)
    leg_low = pd.Series(leg_low, index=close.index)
    support = pd.Series(support, index=close.index)
    fib_zone = retracement.between(FIB_LOW, FIB_HIGH)

    macd_line, signal_line = _macd(close)
    rsi = _rsi(close)
    macd_up = (macd_line > signal_line) & ~(macd_line.shift(1) > signal_line.shift(1))
    rsi_up = (rsi > RSI_TRIGGER) & ~(rsi.shift(1) > RSI_TRIGGER)
    setup = uptrend & fib_zone & support
    recent_setup = setup.astype(float).rolling(SETUP_WINDOW, min_periods=1).max() > 0
    middle = close.rolling(BOLLINGER_DAYS).mean()
    lower = middle - BOLLINGER_SD * close.rolling(BOLLINGER_DAYS).std(ddof=0)
    touched = (close <= lower).astype(float).rolling(SETUP_WINDOW, min_periods=1).max() > 0
    bollinger = touched & (close > lower)
    trigger = (macd_up | rsi_up) & recent_setup & bollinger
    valid = ma[MA_SLOW].notna()
    frame = pd.DataFrame({"uptrend": uptrend, "fib_zone": fib_zone, "support": support,
                          "bollinger": bollinger, "trigger": trigger, "setup": setup,
                          "retracement": retracement, "leg_low": leg_low,
                          "ma_slow": ma[MA_SLOW], "close": close})
    return frame[valid]


@dataclass(frozen=True)
class PositionSignals:
    signals: StrategySignals
    conditions: pd.DataFrame = field(repr=False)
    stops: pd.Series = field(repr=False)  # the stop in force each day (NaN when flat)

    @property
    def open_at_end(self) -> bool:
        """The last trade has not been exited: it is marked at the last price."""
        e, x = self.signals.entries, self.signals.exits
        last_in = e[e].index.max() if e.any() else None
        last_out = x[x].index.max() if x.any() else None
        return last_in is not None and (last_out is None or last_in > last_out)

    def condition_days(self) -> dict[str, int]:
        """Days each condition held, and days all held (for 'what blocked trades')."""
        frame = self.conditions
        out = {name: int(frame[name].sum()) for name in CONDITIONS}
        out["all"] = int((frame["uptrend"] & frame["trigger"]).sum())
        out["days"] = len(frame)
        return out


def position_signals(ohlc: pd.DataFrame, symbol: str = "",
                     state: SwingState | None = None) -> PositionSignals:
    """Entries and exits for the rule; the stop is path-dependent, so this walks the days."""
    state = state or swing_state(weekly_swings(ohlc))
    frame = conditions(ohlc, state)
    entries = pd.Series(False, index=ohlc.index)
    exits = pd.Series(False, index=ohlc.index)
    stops = pd.Series(np.nan, index=ohlc.index)
    in_trade, stop = False, math.nan
    for day, row in frame.iterrows():
        price = row["close"]
        if not in_trade:
            # The pullback may have bounced up out of the zone by the time momentum
            # turns, but not broken down through it: deeper than 61.8% it failed.
            intact = math.isfinite(row["retracement"]) and row["retracement"] <= FIB_HIGH
            if row["uptrend"] and row["trigger"] and intact \
                    and math.isfinite(row["leg_low"]) and row["leg_low"] < price:
                in_trade, stop = True, float(row["leg_low"])
                entries.loc[day] = True
                stops.loc[day] = stop
            continue
        lows = state.lows_known(day)
        higher = lows[(lows > stop) & (lows < price)]
        if higher.size:  # raise the stop to the highest confirmed swing low below price
            stop = float(higher.max())
        stops.loc[day] = stop
        if price < stop or price < row["ma_slow"]:
            exits.loc[day] = True
            in_trade, stop = False, math.nan
    rationale = ("Pullback in an uptrend (ThinkMarkets position-trading guide): enter when "
                 + "; ".join(CONDITION_TEXT[c].lower() for c in CONDITIONS)
                 + f". Exit on a close below the stop (the leg's swing low, raised to each "
                 f"higher weekly swing low) or below the {MA_SLOW}-day average.")
    return PositionSignals(StrategySignals(symbol, entries, exits, rationale), frame, stops)


# --------------------------------------------------------------------------
# Today's levels (shown in the report)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Levels:
    close: float
    moving_averages: dict[int, float]
    bollinger: tuple[float, float, float]  # lower, middle, upper
    percent_b: float
    leg: Leg | None
    retracement: float
    support: float | None  # nearest support at or below the close
    resistance: float | None  # nearest confirmed weekly swing high above the close
    conditions_now: dict[str, bool]

    def fib_levels(self) -> dict[str, float]:
        if self.leg is None:
            return {}
        out = {f"{f:.1%}": self.leg.level(f) for f in (0.382, FIB_LOW, FIB_HIGH)}
        out.update({f"{m:.1%} ext.": self.leg.extension(m) for m in FIB_EXTENSIONS})
        return out


def levels(ohlc: pd.DataFrame, state: SwingState | None = None,
           frame: pd.DataFrame | None = None) -> Levels:
    """Today's levels; pass the state and conditions already computed to skip redoing them."""
    close = ohlc["Close"].astype(float)
    price = float(close.iloc[-1])
    day = close.index[-1]
    mas = {n: float(close.rolling(n).mean().iloc[-1]) for n in (MA_FAST, MA_MID, MA_SLOW)}
    mid = close.rolling(BOLLINGER_DAYS).mean()
    sd = close.rolling(BOLLINGER_DAYS).std(ddof=0)
    lower = float(mid.iloc[-1] - BOLLINGER_SD * sd.iloc[-1])
    upper = float(mid.iloc[-1] + BOLLINGER_SD * sd.iloc[-1])
    state = state or swing_state(weekly_swings(ohlc))
    leg = state.leg(day)
    lows, highs = state.lows_known(day), state.highs_known(day)
    supports = [float(v) for v in lows if v <= price]
    supports += [mas[n] for n in (MA_MID, MA_SLOW) if math.isfinite(mas[n]) and mas[n] <= price]
    resistances = [float(v) for v in highs if v > price]
    frame = conditions(ohlc, state) if frame is None else frame
    now = ({name: bool(frame[name].iloc[-1]) for name in CONDITIONS} if len(frame)
           else {name: False for name in CONDITIONS})
    return Levels(
        close=price, moving_averages=mas, bollinger=(lower, float(mid.iloc[-1]), upper),
        percent_b=(price - lower) / (upper - lower) if upper > lower else float("nan"),
        leg=leg, retracement=leg.retracement(price) if leg else float("nan"),
        support=max(supports) if supports else None,
        resistance=min(resistances) if resistances else None, conditions_now=now)


# --------------------------------------------------------------------------
# Backtest with Monte Carlo, for the report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionBacktest:
    signals: PositionSignals = field(repr=False)
    backtest: object  # backtests.backtest_engine.BacktestResult
    monte_carlo: object | None  # backtests.monte_carlo.MonteCarloResult
    simulated_returns: object = field(default=None, repr=False)
    levels: Levels | None = None

    @property
    def stop_in_force(self) -> float | None:
        """The trailing stop of a trade still open at the last bar, else None."""
        stops = self.signals.stops.dropna()
        return float(stops.iloc[-1]) if self.signals.open_at_end and len(stops) else None


def run(symbol: str, history: pd.DataFrame, num_simulations: int = 2000,
        seed: int = 42) -> PositionBacktest:
    """Backtest the rule on `history` (daily OHLCV) and resample its returns."""
    from backtests import backtest_engine, monte_carlo

    ohlc = history[["Open", "High", "Low", "Close"]].astype(float)
    state = swing_state(weekly_swings(ohlc))
    sig = position_signals(ohlc, symbol, state)
    result = backtest_engine.run_backtest(history, sig.signals.entries, sig.signals.exits,
                                          fees_bps=FEES_BPS, symbol=symbol)
    equity = result.equity_curve.to_numpy()
    returns = np.diff(equity) / equity[:-1]
    simulation = simulated = None
    if result.num_trades and returns.size > 2 and returns.std() > 0:
        simulation = monte_carlo.run_monte_carlo(
            symbol, returns, num_simulations=num_simulations,
            observed_return_pct=result.annualized_return_pct, seed=seed)
        paths = monte_carlo.resample_returns(returns, num_simulations, 252, seed=seed)
        simulated = (np.cumprod(1.0 + paths, axis=1)[:, -1] - 1.0) * 100.0
    return PositionBacktest(sig, result, simulation, simulated,
                            levels(ohlc, state, sig.conditions))
