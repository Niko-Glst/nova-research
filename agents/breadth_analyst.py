"""Market breadth: the equal-weight S&P 500 over the cap-weight S&P 500 (RSP / SPY).

When the ratio rises, the average stock is outrunning the largest ones: the
market is broadening. When it falls, a few mega caps carry the index: it is
narrowing. The ratio moves in long waves because the flows behind it do:
pension and asset-manager rebalancing, hedge-fund positioning, automatic
index and target-weight rebalancing. It also tends to respect its own trend.

Rule (owner's decisions, 2026-10-25; numbers in config/scoring.yaml):

- Position in the 1-year range of the ratio: 0 at its low, 1 at its high.
- **Bottom zone** (lowest 20%) and the ratio back above its 50-day average:
  the narrowing is exhausted, so favour breadth (+1). In the zone without
  that turn: be ready, no call (regime 0, and the leg is left out of the
  verdict rather than counted as neutral).
- **Top zone** (highest 20%) and the ratio back below its 50-day average:
  the broadening is tired, so favour the leaders (-1). Without the turn: no call.
- **Between the zones** the local trend decides: the slope of a least-squares
  trend line through ln(ratio) over the last ~3 months. The ratio zigzags,
  but a few months of it show the direction: rising favours breadth (+1),
  falling favours the leaders (-1).

The regime is then read for the stock:

- *breadth* stocks (Healthcare, Financial Services, Consumer Defensive,
  Utilities, Industrials, Real Estate) score the regime as it is;
- *leaders* (Technology, Communication Services and Consumer Cyclical with a
  market cap of $500B or more: the mega caps) score its opposite;
- any other stock gets no breadth score, so the leg is left out of its
  verdict and the other legs are reweighted.

This is a reading of conditions. It was weighted 0.8 at first and lowered to
0.5 after its backtest (history.py breadth-backtest) found no significant edge.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from agents.ta_analyst import fetch_price_history
from common.rate_limit import get_limiter
from config.settings import load_scoring_config

_CFG = load_scoring_config()["breadth"]
EQUAL_WEIGHT = str(_CFG["equal_weight"])
CAP_WEIGHT = str(_CFG["cap_weight"])
PRICE_PERIOD = str(_CFG["price_period"])
RANGE_DAYS = int(_CFG["range_days"])
ZONE = float(_CFG["zone"])
TURN_MA = int(_CFG["turn_ma_days"])
TREND_DAYS = int(_CFG["trend_days"])
TREND_LABEL = f"{round(TREND_DAYS / 21)}-month trend line"  # 21 trading days a month
CHART_VIEWS = {str(k): int(v) for k, v in _CFG["chart_views"].items()}
ZONE_TURN_SCORE = float(_CFG["zone_turn_score"])
ZONE_WAITING_SCORE = float(_CFG["zone_waiting_score"])
TREND_SCORE = float(_CFG["trend_score"])
LEADER_SECTORS = tuple(_CFG["leader_sectors"])
LEADER_MIN_CAP = float(_CFG["leader_min_market_cap"])
BREADTH_SECTORS = tuple(_CFG["breadth_sectors"])

LEADER, BREADTH, NEUTRAL = "leader", "breadth", "neutral"

_STATE_TEXT = {
    "bottom_turn": "in the bottom zone and back above its {turn}-day average: the narrowing "
                   "looks exhausted, favour breadth",
    "bottom_waiting": "in the bottom zone but not yet back above its {turn}-day average: "
                      "be ready for breadth, no call yet",
    "top_turn": "in the top zone and back below its {turn}-day average: the broadening looks "
                "tired, favour the leaders",
    "top_waiting": "in the top zone but not yet back below its {turn}-day average: be ready "
                   "to narrow, no call yet",
    "trend_broadening": "between the zones, and its {trend}-day trend line is rising: "
                        "broadening, favour breadth",
    "trend_narrowing": "between the zones, and its {trend}-day trend line is falling: "
                       "narrowing, favour the leaders",
}


@dataclass(frozen=True)
class BreadthRegime:
    ratio: float
    low: float
    high: float
    position: float  # 0 = 1-year low, 1 = 1-year high
    ma_turn: float
    trend_slope: float  # ln(ratio) per trading day, over TREND_DAYS
    regime: float  # +1 favours breadth, -1 the leaders, 0 no call
    state: str

    @property
    def trend_change(self) -> float:
        """The trend line's change over its window, as a fraction."""
        return math.expm1(self.trend_slope * (TREND_DAYS - 1))

    def describe(self) -> str:
        text = _STATE_TEXT[self.state].format(turn=TURN_MA, trend=TREND_DAYS)
        return (f"{EQUAL_WEIGHT}/{CAP_WEIGHT} {self.ratio:.4f} sits at {self.position:.0%} of "
                f"its {RANGE_DAYS}-day range ({self.low:.4f} to {self.high:.4f}), {text} "
                f"({TURN_MA}-day average {self.ma_turn:.4f}; trend line "
                f"{self.trend_change:+.1%} over {TREND_DAYS} days).")


@dataclass(frozen=True)
class BreadthRead:
    symbol: str
    signal: str  # "bullish" / "neutral" / "bearish" / "unknown"
    score: float | None
    group: str  # LEADER, BREADTH or NEUTRAL
    sector: str = ""
    market_cap: float | None = None
    regime: BreadthRegime | None = None
    reasons: list[str] = field(default_factory=list)
    # The ratio over the longest chart view plus the turn average's warm-up.
    history: pd.Series = field(default_factory=pd.Series, repr=False)


def ratio_series(equal_weight: pd.Series, cap_weight: pd.Series) -> pd.Series:
    """RSP / SPY on the dates both traded."""
    # sort=True: the last row must be the latest date (pandas 3 stops sorting by default).
    frame = pd.concat({"ew": equal_weight, "cw": cap_weight}, axis=1, sort=True).dropna()
    return (frame["ew"] / frame["cw"]).rename("ratio")


def trend_line(values: pd.Series, days: int = TREND_DAYS) -> tuple[float, np.ndarray]:
    """Slope (ln per day) and fitted ratio of a least-squares line through ln(ratio)."""
    window = np.log(values.dropna().to_numpy(dtype=float)[-days:])
    x = np.arange(len(window), dtype=float)
    slope, intercept = np.polyfit(x, window, 1)
    return float(slope), np.exp(intercept + slope * x)


def regime_from_ratio(ratio: pd.Series) -> BreadthRegime | None:
    """The regime at the last date of `ratio`, using only that history."""
    values = ratio.dropna()
    if len(values) < max(RANGE_DAYS, TREND_DAYS, TURN_MA):
        return None
    now = float(values.iloc[-1])
    window = values.iloc[-RANGE_DAYS:]
    low, high = float(window.min()), float(window.max())
    if not high > low:
        return None
    position = (now - low) / (high - low)
    ma_turn = float(values.iloc[-TURN_MA:].mean())
    slope, _ = trend_line(values)
    if position <= ZONE:
        state, regime = (("bottom_turn", ZONE_TURN_SCORE) if now > ma_turn
                         else ("bottom_waiting", ZONE_WAITING_SCORE))
    elif position >= 1.0 - ZONE:
        state, regime = (("top_turn", -ZONE_TURN_SCORE) if now < ma_turn
                         else ("top_waiting", -ZONE_WAITING_SCORE))
    elif slope > 0:
        state, regime = "trend_broadening", TREND_SCORE
    else:
        state, regime = "trend_narrowing", -TREND_SCORE
    return BreadthRegime(now, low, high, position, ma_turn, slope, regime + 0.0, state)


def classify_stock(sector: str, market_cap: float | None) -> str:
    if sector in BREADTH_SECTORS:
        return BREADTH
    if sector in LEADER_SECTORS and market_cap is not None and market_cap >= LEADER_MIN_CAP:
        return LEADER
    return NEUTRAL


def score_for(group: str, regime: float) -> float | None:
    """The stock's breadth score, or None: no group, or no call (regime 0).

    In a zone without the turn the regime is 0, which is waiting, not a reading
    of balance; the owner (2026-10-25) leaves the leg out then rather than
    letting a neutral pull the verdict toward "mixed".
    """
    if regime == 0:
        return None
    if group == BREADTH:
        return regime
    if group == LEADER:
        return -regime
    return None


def stance(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return "unknown"
    return "bullish" if score > 0 else "bearish" if score < 0 else "neutral"


def _group_text(group: str, sector: str, market_cap: float | None) -> str:
    cap = f", market cap ${market_cap / 1e9:,.0f}B" if market_cap else ""
    if group == BREADTH:
        return f"{sector or 'its sector'} is a breadth sector{cap}: it scores the regime as is."
    if group == LEADER:
        return (f"A mega-cap leader ({sector}{cap}): it scores the opposite of the regime.")
    return (f"Neither a breadth sector nor a mega-cap leader ({sector or 'sector unknown'}{cap}): "
            f"no breadth score; the leg is left out of this verdict.")


def analyze(symbol: str, sector: str | None = None,
            market_cap: float | None = None) -> BreadthRead:
    """The breadth regime and what it means for `symbol`."""
    symbol = symbol.strip().upper()
    if sector is None or market_cap is None:
        get_limiter("yfinance").wait()
        info = yf.Ticker(symbol).info or {}
        sector = sector if sector is not None else str(info.get("sector") or "")
        market_cap = market_cap if market_cap is not None else info.get("marketCap")
    group = classify_stock(sector, market_cap)
    ew = fetch_price_history(EQUAL_WEIGHT, period=PRICE_PERIOD)["Close"]
    cw = fetch_price_history(CAP_WEIGHT, period=PRICE_PERIOD)["Close"]
    ratio = ratio_series(ew, cw)
    index = pd.DatetimeIndex(ratio.index)
    ratio.index = index.tz_localize(None) if index.tz is not None else index
    history = ratio.iloc[-(max(CHART_VIEWS.values()) + TURN_MA):]
    regime = regime_from_ratio(ratio)
    reasons = [_group_text(group, sector, market_cap)]
    if regime is None:
        reasons.insert(0, f"Not enough {EQUAL_WEIGHT}/{CAP_WEIGHT} history for the range.")
        return BreadthRead(symbol, "unknown", None, group, sector, market_cap, None, reasons,
                           history)
    reasons.insert(0, regime.describe())
    score = score_for(group, regime.regime)
    if regime.regime == 0 and group != NEUTRAL:
        reasons.append("No call while the ratio waits in the zone: the leg is left out of "
                       "this verdict.")
    return BreadthRead(symbol, stance(score), score, group, sector, market_cap, regime, reasons,
                       history)
