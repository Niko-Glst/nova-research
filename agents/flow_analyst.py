"""Market-flow analyst: who is likely to be buying or selling, from public data.

Prices move on flows as well as on fundamentals. Four flows matter most:
corporate buybacks, systematic (trend-following and volatility-targeting)
funds, options dealers hedging, and real-money rebalancing. Public data shows
the *conditions* for these flows, not the positions behind them, so every
figure here is a proxy and says so.

Choices made with the owner on 2026-10-21:

- This is a scored leg, weight 0.4 (research_report.WEIGHTS), equal to
  sentiment: the signals are proxies and none has been tested yet.
- Systematic funds: a market backdrop (the S&P 500 via SPY) plus the stock's
  own volatility. The stock's *trend* is left to ta_analyst so it is not
  counted twice.
- Options: raw data only. Who is long or short is not in the data, so no
  dealer-gamma estimate is made.
- Real-money rebalancing is left out; institutional ownership is shown as a fact.

Scored parts, each in [-1, 1] with its reason, or unknown with the reason
(model 2.0.0, 2026-10-22; thresholds and weights in config/scoring.yaml):

1. **Net issuance** (Fama-French): ln(shares at fiscal year-end t-1) - ln(shares
   at t-2), split-adjusted. Primary source: the latest SEC 10-K, both counts
   from the same filing (US filers); otherwise yfinance's annual balance
   sheet. Issuing shares scores down to -1, shrinking the count up to +0.5:
   the issuance side is weighted more, as the anomaly is driven by issuers.
   Cash-flow lines confirm it, flagged per ticker: reported, assumed_zero
   (statement present, buyback line absent), confirmed_zero (the 10-K has no
   repurchase fact), unknown (no statement; the share count still scores), or
   data_conflict (no buyback line, yet the count fell: this part is dropped).
2. **Systematic funds, market backdrop.** The S&P 500's trend over 1, 3 and 12
   months (the horizons of Hurst, Ooi and Pedersen's "A Century of Evidence on
   Trend-Following Investing") and its volatility over the last month against
   the last year. Trend up on at least two horizons with volatility at or below
   its norm is the setting in which these funds add exposure (+1); trend down
   with volatility clearly rising is the setting in which they cut it (-1).
3. **Systematic funds, the stock.** Its volatility over the last month against
   the last year: settling (+1) or rising (-1). Volatility-based sizing holds
   more of an asset as its volatility falls.

Stance = mean of the parts that could be scored: bullish at or above
+STANCE_THRESHOLD, bearish at or below minus it, neutral between; unknown when
no part could be scored. Shown but not scored: options open interest and
implied volatility, ownership percentages, and buybacks quarter by quarter
next to that quarter's price move (a description of a handful of quarters,
not a statistic).

Every threshold was set by judgement with the owner, not fitted: none of these
signals has a point-in-time history to test against yet.
agents/signal_snapshots.py records them on every run so they can be tested
later.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from types import MappingProxyType

import numpy as np
import pandas as pd
import yfinance as yf

from agents import currency, sec_edgar, volatility as vol
from agents.risk_manager import TRADING_DAYS
from agents.ta_analyst import fetch_price_history
from common.rate_limit import get_limiter
from config.settings import load_scoring_config

# Every threshold below comes from config/scoring.yaml, where each is explained.
_FLOWS = load_scoring_config()["flows"]
_MARKET = _FLOWS["market"]
_STOCK = _FLOWS["stock_volatility"]
_OPTIONS = _FLOWS["options"]

MARKET_PROXY = str(_MARKET["proxy"])
PRICE_PERIOD = str(_MARKET["price_period"])
TREND_HORIZONS = tuple(int(h) for h in _MARKET["trend_horizons"])
TREND_MIN_VOTES = int(_MARKET["trend_min_votes"])
VOL_SHORT, VOL_LONG = int(_MARKET["vol_short"]), int(_MARKET["vol_long"])
MARKET_CALM_RATIO = float(_MARKET["calm_ratio"])
RISING_VOL_RATIO = float(_MARKET["rising_ratio"])

STOCK_SCORING = str(_STOCK["scoring"])  # "continuous" or "discrete"
STOCK_SHORT, STOCK_LONG = int(_STOCK["short_window"]), int(_STOCK["long_window"])
STOCK_SETTLING_RATIO = float(_STOCK["settling_ratio"])
STOCK_RISING_RATIO = float(_STOCK["rising_ratio"])

STANCE_THRESHOLD = float(_FLOWS["stance_threshold"])
COMPONENT_WEIGHTS = {str(k): float(v) for k, v in _FLOWS["component_weights"].items()}
# Parts about the stock itself; the leg needs one of them to take a stance.
STOCK_PARTS = ("issuance", "systematic_stock")
DEFAULT_COMPONENT_WEIGHT = float(_FLOWS["default_component_weight"])
_ISSUANCE = _FLOWS["issuance"]
ISSUANCE_FULL_SCALE = float(_ISSUANCE["full_scale"])
ISSUANCE_SIDE_WEIGHT = float(_ISSUANCE["side_weights"]["issuance"])
BUYBACK_SIDE_WEIGHT = float(_ISSUANCE["side_weights"]["buyback"])
CONFLICT_TOLERANCE = float(_ISSUANCE["conflict_tolerance"])
SPLIT_MATCH_TOLERANCE = float(_ISSUANCE["split_match_tolerance"])
QUARTERS_SHOWN = int(_ISSUANCE["quarters_shown"])
QUARTER_PRICE_GAP_DAYS = int(_ISSUANCE["quarter_price_gap_days"])

OPTIONS_HORIZON_DAYS = int(_OPTIONS["horizon_days"])
MAX_OPTION_EXPIRATIONS = int(_OPTIONS["max_expirations"])
MIN_IV_DAYS = int(_OPTIONS["min_iv_days"])
IV_PLAUSIBLE = tuple(float(v) for v in _OPTIONS["iv_plausible"])

# yfinance cash-flow row labels (identifiers, not thresholds).
REPURCHASE_ROW = "Repurchase Of Capital Stock"
ISSUANCE_ROW = "Issuance Of Capital Stock"


@dataclass(frozen=True)
class FlowComponent:
    name: str  # "issuance", "systematic_market", "systematic_stock"
    label: str
    score: float | None  # in [-1, 1]; None when it could not be computed
    reason: str
    values: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    flag: str = ""  # data-quality or evidence flag shown in the report


@dataclass(frozen=True)
class QuarterBuyback:
    quarter_end: date
    repurchased: float  # dollars spent buying back shares (positive)
    issued: float  # dollars raised selling new shares (positive)
    price_return: float | None  # the stock's return over that quarter


@dataclass(frozen=True)
class OptionsSnapshot:
    expirations: tuple[str, ...]
    call_open_interest: int
    put_open_interest: int
    put_call_ratio: float
    top_call_strikes: tuple[tuple[float, int], ...]  # (strike, open interest)
    top_put_strikes: tuple[tuple[float, int], ...]
    atm_implied_vol: float  # annualised; NaN when no plausible quote
    atm_expiration: str
    realised_vol: float  # the stock's last-month volatility, annualised


@dataclass(frozen=True)
class FlowRead:
    symbol: str
    signal: str  # "bullish" / "neutral" / "bearish" / "unknown"
    score: float  # mean of the scored parts; NaN when none could be scored
    components: tuple[FlowComponent, ...]
    reasons: list[str] = field(default_factory=list)
    options: OptionsSnapshot | None = None
    options_note: str = ""
    quarters: tuple[QuarterBuyback, ...] = ()
    institutions_pct: float | None = None
    insiders_pct: float | None = None
    market_symbol: str = MARKET_PROXY
    notes: str = ""
    report_currency: str = "USD"  # currency of the quarterly buyback amounts

    def component(self, name: str) -> FlowComponent | None:
        return next((c for c in self.components if c.name == name), None)


# --------------------------------------------------------------------------
# Pure calculations (tested on synthetic data)
# --------------------------------------------------------------------------


def _naive(series: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return pd.Series(series.to_numpy(dtype=float), index=index).sort_index()


def trend_returns(close: pd.Series, horizons=TREND_HORIZONS) -> dict[int, float]:
    """Simple return over each horizon (trading days); NaN when the history is shorter."""
    values = close.dropna().to_numpy(dtype=float)
    out = {}
    for h in horizons:
        out[h] = float(values[-1] / values[-1 - h] - 1.0) if len(values) > h else float("nan")
    return out


def volatility_ratio(close: pd.Series, short: int = VOL_SHORT, long: int = VOL_LONG) -> float:
    """Std of the last `short` daily log returns over the std of the last `long`."""
    returns = np.diff(np.log(close.dropna().to_numpy(dtype=float)))
    if len(returns) < long:
        return float("nan")
    long_std = returns[-long:].std(ddof=1)
    return float(returns[-short:].std(ddof=1) / long_std) if long_std > 0 else float("nan")


def realised_volatility(close: pd.Series, window: int = VOL_SHORT) -> float:
    returns = np.diff(np.log(close.dropna().to_numpy(dtype=float)))
    if len(returns) < window:
        return float("nan")
    return float(returns[-window:].std(ddof=1) * math.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------
# Scoring rules as pure functions: the live components and history.py rescore
# call the same code, so a rescored run is scored exactly as a live one.
# --------------------------------------------------------------------------


def market_score(returns: tuple[float, ...] | list[float], ratio: float) -> float:
    """+1 when the trend is up on enough horizons and volatility is calm, -1 for the
    reverse with volatility clearly rising, otherwise 0."""
    ups = sum(r > 0 for r in returns)
    downs = sum(r < 0 for r in returns)
    if ups >= TREND_MIN_VOTES and ratio <= MARKET_CALM_RATIO:
        return 1.0
    if downs >= TREND_MIN_VOTES and ratio >= RISING_VOL_RATIO:
        return -1.0
    return 0.0


def stock_score(z: float, ratio: float, scoring: str | None = None) -> float:
    """Continuous: -clip(z, -1, 1). Discrete (1.1.0): +1 / 0 / -1 on the ratio thresholds."""
    if (scoring or STOCK_SCORING) == "discrete":
        if ratio <= STOCK_SETTLING_RATIO:
            return 1.0
        return -1.0 if ratio >= STOCK_RISING_RATIO else 0.0
    return -float(np.clip(z, -1.0, 1.0))


def issuance_score(net_issuance: float) -> float:
    """Issuing scores down to -ISSUANCE_SIDE_WEIGHT, shrinking up to +BUYBACK_SIDE_WEIGHT."""
    if net_issuance > 0:
        return -min(net_issuance / ISSUANCE_FULL_SCALE, 1.0) * ISSUANCE_SIDE_WEIGHT
    return min(-net_issuance / ISSUANCE_FULL_SCALE, 1.0) * BUYBACK_SIDE_WEIGHT


def is_data_conflict(net_issuance: float, buyback_flag: str) -> bool:
    """No buyback line (assumed or confirmed zero), yet the share count fell."""
    return buyback_flag in ("assumed_zero", "confirmed_zero") and \
        net_issuance < -CONFLICT_TOLERANCE


def stance_from_score(score: float) -> str:
    if not math.isfinite(score):
        return "unknown"
    if score >= STANCE_THRESHOLD:
        return "bullish"
    return "bearish" if score <= -STANCE_THRESHOLD else "neutral"


def weighted_score(scores, weights: dict[str, float] | None = None) -> float:
    """Sum of weight x score over the parts that could be scored, divided by their
    total weight: always in [-1, 1], and a missing part is reweighted away, never
    counted as 0. `scores` is a mapping or (name, score) pairs (names may repeat)."""
    weights = COMPONENT_WEIGHTS if weights is None else weights
    items = scores.items() if isinstance(scores, Mapping) else scores
    pairs = [(weights.get(name, DEFAULT_COMPONENT_WEIGHT), value)
             for name, value in items if value is not None and math.isfinite(value)]
    total = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total if total > 0 else float("nan")


def market_component(close: pd.Series, symbol: str = MARKET_PROXY) -> FlowComponent:
    """Systematic funds' setting from the market's trend and volatility."""
    label = f"Systematic funds: market ({symbol})"
    trend = trend_returns(close)
    ratio = volatility_ratio(close)
    if any(math.isnan(v) for v in trend.values()) or math.isnan(ratio):
        return FlowComponent("systematic_market", label, None,
                             f"Not enough {symbol} history for a 12-month trend and volatility.")
    ups = sum(v > 0 for v in trend.values())
    downs = sum(v < 0 for v in trend.values())
    values = MappingProxyType({
        "return_1m": trend[21], "return_3m": trend[63], "return_12m": trend[252],
        "up_horizons": float(ups), "vol_ratio": ratio,
    })
    moves = (f"{trend[21]:+.2%} over 1 month, {trend[63]:+.1%} over 3, "
             f"{trend[252]:+.1%} over 12")
    score = market_score(tuple(trend.values()), ratio)
    if score > 0:
        return FlowComponent("systematic_market", label, score, (
            f"{symbol} is up on {ups} of 3 horizons ({moves}) and its volatility is "
            f"{ratio:.2f}x its one-year norm: the setting in which trend-following and "
            f"volatility-targeting funds add exposure."), values)
    if score < 0:
        return FlowComponent("systematic_market", label, score, (
            f"{symbol} is down on {downs} of 3 horizons ({moves}) and its volatility is "
            f"{ratio:.2f}x its one-year norm: the setting in which these funds cut exposure."),
            values)
    return FlowComponent("systematic_market", label, score, (
        f"{symbol}: {moves}, volatility {ratio:.2f}x its one-year norm. No clear push from "
        f"systematic funds either way."), values)


_ESTIMATOR_TEXT = {
    vol.YANG_ZHANG: "Yang-Zhang on daily open, high, low and close",
    vol.CC_FALLBACK: "close-to-close: the OHLC bars failed the quality check",
    vol.CLOSE_TO_CLOSE: "close-to-close: no OHLC bars",
}


def stock_volatility_component(
    prices: pd.DataFrame | pd.Series, symbol: str, scoring: str | None = None
) -> FlowComponent:
    """Volatility regime of the stock itself (its trend is ta_analyst's job).

    `prices` is an OHLC frame (or a close series, which forces close-to-close).
    Scoring per config: "continuous" gives -clip(z, -1, 1) with
    z = ln(short_vol / long_vol) / SE; "discrete" is the 1.1.0 rule.
    """
    scoring = scoring or STOCK_SCORING
    label = "Systematic funds: the stock's volatility"
    ohlc = prices.to_frame("Close") if isinstance(prices, pd.Series) else prices
    reading = vol.volatility_ratio(ohlc, STOCK_SHORT, STOCK_LONG)
    flag = f"vol_estimator: {reading.estimator}"
    if not math.isfinite(reading.ratio):
        return FlowComponent("systematic_stock", label, None,
                             f"Not enough {symbol} history for a one-year volatility.", flag=flag)
    ratio = reading.ratio
    values = MappingProxyType({
        "vol_ratio": ratio, "z": reading.z, "se": reading.se,
        "short_vol": reading.short_vol * math.sqrt(TRADING_DAYS),
        "long_vol": reading.long_vol * math.sqrt(TRADING_DAYS),
        "realised_vol": realised_volatility(ohlc["Close"]),
        "bad_row_share": reading.bad_row_share,
    })
    score = stock_score(reading.z, ratio, scoring)
    if scoring == "continuous" and math.isfinite(reading.z):
        # Worded from z, as it is scored; the ratio bands below belong to the
        # discrete rule (a 0.83 ratio at z -2.5 is a clear fall, scored +1).
        size = f"{abs(reading.z):.1f} standard errors"
        reading_text = ("below its norm by " + size + ", so volatility-based sizing can hold "
                        "more of it" if reading.z < 0 else
                        "above its norm by " + size + ", so volatility-based sizing holds "
                        "less of it" if reading.z > 0 else "at its norm")
    elif ratio <= STOCK_SETTLING_RATIO:
        reading_text = "settling, so volatility-based sizing can hold more of it"
    elif ratio >= STOCK_RISING_RATIO:
        reading_text = "rising, so volatility-based sizing holds less of it"
    else:
        reading_text = "within the band where no clear change is called"
    quality = (f"; {reading.bad_row_share:.0%} of bars suspect"
               if reading.estimator == vol.CC_FALLBACK and math.isfinite(reading.bad_row_share)
               else "")
    return FlowComponent("systematic_stock", label, score, (
        f"Volatility over the last {STOCK_SHORT} days is {ratio:.2f}x its {STOCK_LONG}-day "
        f"norm (z {reading.z:+.2f}, {_ESTIMATOR_TEXT[reading.estimator]}{quality}): "
        f"{reading_text}. Score {score:+.2f} ({scoring})."), values, flag)


# Buyback-line evidence flags (owner 2026-10-22), shown per ticker in the report.
REPORTED = "reported"  # the repurchase line is there, with a value
ASSUMED_ZERO = "assumed_zero"  # statement present, line absent: counted as no buybacks
CONFIRMED_ZERO = "confirmed_zero"  # the SEC 10-K for the year has no repurchase fact
UNKNOWN = "unknown"  # no cash-flow statement for the period
DATA_CONFLICT = "data_conflict"  # no buyback line, yet the share count fell

_FLAG_TEXT = {
    REPORTED: "reported in the cash-flow statement",
    ASSUMED_ZERO: "no buyback line in the statement, counted as none (assumed_zero)",
    CONFIRMED_ZERO: "the 10-K has no repurchase fact for the year (confirmed_zero)",
    UNKNOWN: "no cash-flow statement for the year, so nothing confirms it (unknown)",
}


@dataclass(frozen=True)
class IssuanceInputs:
    """Share counts and cash-flow evidence for the last completed fiscal year."""

    net_issuance: float | None  # ln(shares at year-end t-1 / shares at t-2)
    year_end: date | None
    source: str  # where the share counts came from
    buyback_flag: str  # REPORTED / ASSUMED_ZERO / CONFIRMED_ZERO / UNKNOWN
    repurchased: float | None = None  # over the year, in `currency`
    issued: float | None = None
    note: str = ""
    currency: str = "USD"  # the statements' currency


def issuance_component(inputs: IssuanceInputs, market_cap: float | None = None) -> FlowComponent:
    """Net issuance (Fama-French) scored with the issuance side weighted more."""
    label = "Net issuance (share count)"
    ni = inputs.net_issuance
    values: dict[str, float] = {}
    for key, value in (("repurchased", inputs.repurchased), ("issued", inputs.issued)):
        if value is not None:
            values[key] = float(value)
            if market_cap:
                values[f"{key}_yield"] = float(value) / float(market_cap)
    if ni is None or not math.isfinite(ni):
        return FlowComponent("issuance", label, None, (
            f"No share counts for the last two fiscal year-ends ({inputs.source})."
            + (f" {inputs.note}" if inputs.note else "")),
            MappingProxyType(values), inputs.buyback_flag or UNKNOWN)
    values.update(net_issuance=ni)
    year = inputs.year_end.isoformat() if inputs.year_end else "the last fiscal year"
    if is_data_conflict(ni, inputs.buyback_flag):
        return FlowComponent("issuance", label, None, (
            f"Shares fell {-ni:.1%} over the year to {year} ({inputs.source}), but "
            f"{_FLAG_TEXT[inputs.buyback_flag]}. The two disagree, so this part is left out."),
            MappingProxyType(values), DATA_CONFLICT)
    score = issuance_score(ni) + 0.0  # never -0.0
    if ni > 0:
        direction = (f"shares grew: issuance adds supply (issuance side, weight "
                     f"{ISSUANCE_SIDE_WEIGHT:g})")
    elif ni == 0:
        direction = "share count unchanged: no net supply or demand"
    else:
        direction = (f"shares shrank: repurchases add demand (buyback side, weight "
                     f"{BUYBACK_SIDE_WEIGHT:g})")
    amounts = []
    sign = currency.prefix(inputs.currency)
    if inputs.repurchased is not None:
        amounts.append(f"{sign}{inputs.repurchased / 1e6:,.0f}M bought back")
    if inputs.issued is not None:
        amounts.append(f"{sign}{inputs.issued / 1e6:,.0f}M raised from new shares")
    return FlowComponent("issuance", label, float(score), (
        f"Net issuance {ni:+.1%} over the year to {year} ({inputs.source}): {direction}. "
        + (f"{', '.join(amounts)}. " if amounts else "")
        + f"Buyback line: {_FLAG_TEXT.get(inputs.buyback_flag, inputs.buyback_flag)}."
        + (f" {inputs.note}" if inputs.note else "")),
        MappingProxyType(values), inputs.buyback_flag)


def sec_issuance_inputs(report: sec_edgar.AnnualReport | None) -> IssuanceInputs | None:
    """Inputs from one 10-K: both share counts from the same filing (same split basis)."""
    if report is None:
        return None
    return IssuanceInputs(
        net_issuance=report.net_issuance,
        year_end=report.fiscal_year_end,
        source=f"SEC 10-K filed {report.filed.isoformat()}",
        buyback_flag=REPORTED if report.repurchased is not None else CONFIRMED_ZERO,
        repurchased=report.repurchased,
        issued=report.issued,
    )


def _split_factor(splits: pd.Series | None, after: pd.Timestamp, until: pd.Timestamp) -> float:
    if splits is None or len(splits) == 0:
        return 1.0
    index = pd.DatetimeIndex(splits.index)
    index = index.tz_localize(None) if index.tz is not None else index
    window = splits.to_numpy()[(index > after) & (index <= until)]
    return float(np.prod(window)) if len(window) else 1.0


def yfinance_issuance_inputs(
    balance_sheet: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    splits: pd.Series | None = None,
) -> IssuanceInputs:
    """Inputs from yfinance's annual statements (restated, so live readings only)."""
    source = "yfinance annual balance sheet"
    row = next((r for r in ("Ordinary Shares Number", "Share Issued")
                if balance_sheet is not None and r in balance_sheet.index), None)
    ni, year_end, note = None, None, ""
    if row is not None:
        shares = balance_sheet.loc[row].dropna().astype(float)
        shares.index = pd.DatetimeIndex(shares.index)
        shares = shares[shares > 0].sort_index()
        if len(shares) >= 2:
            (prior_end, prior), (end, current) = list(shares.iloc[-2:].items())
            factor = _split_factor(splits, prior_end, end)
            raw = current / prior
            # yfinance counts are normally split-adjusted already; adjust only when
            # the raw jump matches the split ratio (unadjusted data).
            if factor != 1.0 and abs(raw / factor - 1.0) <= SPLIT_MATCH_TOLERANCE:
                prior *= factor
                note = f"Prior count adjusted for a {factor:g}-for-1 split."
            ni = math.log(current / prior)
            year_end = end.date()
    if cashflow is None or cashflow.empty or year_end is None:
        flag, repurchased, issued = UNKNOWN, None, None
    else:
        column = next((c for c in cashflow.columns if pd.Timestamp(c).date() == year_end), None)
        if column is None:
            flag, repurchased, issued = UNKNOWN, None, None
        else:
            def line(name: str) -> float | None:
                if name not in cashflow.index or pd.isna(cashflow.loc[name, column]):
                    return None
                return float(cashflow.loc[name, column])

            rep, iss = line(REPURCHASE_ROW), line(ISSUANCE_ROW)
            flag = REPORTED if rep is not None else ASSUMED_ZERO
            repurchased = -rep + 0.0 if rep is not None else None
            issued = iss
    return IssuanceInputs(ni, year_end, source, flag, repurchased, issued, note)


def quarterly_buybacks(
    cashflow: pd.DataFrame | None, close: pd.Series | None = None
) -> tuple[QuarterBuyback, ...]:
    """Buybacks and issuance per quarter next to the quarter's price move (shown, not scored)."""
    if cashflow is None or cashflow.empty:
        return ()
    present = [r for r in (REPURCHASE_ROW, ISSUANCE_ROW) if r in cashflow.index]
    if not present:
        return ()
    columns = sorted(cashflow.columns, reverse=True)

    def cell(name: str, col) -> float:
        return float(cashflow.loc[name, col]) if name in cashflow.index else 0.0

    prices = _naive(close) if close is not None and len(close) else None
    quarters = []
    for col in columns[:QUARTERS_SHOWN]:
        end = pd.Timestamp(col)
        rep, iss = cell(REPURCHASE_ROW, col), cell(ISSUANCE_ROW, col)
        if pd.isna(rep) and pd.isna(iss):
            continue  # a blank quarter; one blank line still shows the other (as n/a)
        price_return = None
        if prices is not None:
            start = end - pd.DateOffset(months=3)
            before, after = prices.loc[:start], prices.loc[:end]
            if len(before) and len(after) and before.index[-1] < after.index[-1] \
                    and (start - before.index[-1]).days <= QUARTER_PRICE_GAP_DAYS:
                price_return = float(after.iloc[-1] / before.iloc[-1] - 1.0)
        quarters.append(QuarterBuyback(end.date(), float(-rep) + 0.0 if not pd.isna(rep)
                                       else float("nan"), float(iss), price_return))
    return tuple(quarters)


def options_snapshot(
    chains: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    last_close: float,
    realised_vol: float,
    today: date,
) -> OptionsSnapshot | None:
    """Open interest by side, the busiest strikes, and at-the-money implied volatility."""
    if not chains:
        return None
    expirations = tuple(sorted(chains))
    calls = pd.concat([chains[e][0] for e in expirations], ignore_index=True)
    puts = pd.concat([chains[e][1] for e in expirations], ignore_index=True)
    call_oi = int(calls["openInterest"].fillna(0).sum())
    put_oi = int(puts["openInterest"].fillna(0).sum())

    def top(frame: pd.DataFrame) -> tuple[tuple[float, int], ...]:
        by_strike = frame.groupby("strike")["openInterest"].sum().fillna(0)
        by_strike = by_strike[by_strike > 0].sort_values(ascending=False).head(3)
        return tuple((float(k), int(v)) for k, v in by_strike.items())

    atm_iv, atm_exp = float("nan"), ""
    lo, hi = IV_PLAUSIBLE
    for expiration in expirations:
        if (date.fromisoformat(expiration) - today).days < MIN_IV_DAYS:
            continue
        quotes = []
        for frame in chains[expiration]:
            if frame.empty:
                continue
            nearest = frame.iloc[(frame["strike"] - last_close).abs().argsort()[:1]]
            iv = float(nearest["impliedVolatility"].iloc[0])
            if lo <= iv <= hi:
                quotes.append(iv)
        if quotes:
            atm_iv, atm_exp = float(np.mean(quotes)), expiration
            break
    return OptionsSnapshot(
        expirations=expirations,
        call_open_interest=call_oi,
        put_open_interest=put_oi,
        put_call_ratio=put_oi / call_oi if call_oi else float("nan"),
        top_call_strikes=top(calls),
        top_put_strikes=top(puts),
        atm_implied_vol=atm_iv,
        atm_expiration=atm_exp,
        realised_vol=realised_vol,
    )


def has_stock_part(scores) -> bool:
    """At least one part about the stock itself could be scored.

    The market part describes the market, not the stock: on its own it must not
    give a stock a stance (a ticker with no data at all once read "bullish"
    because the S&P 500 was trending up). `scores` as in weighted_score.
    """
    items = scores.items() if isinstance(scores, Mapping) else scores
    return any(name in STOCK_PARTS and value is not None and math.isfinite(value)
               for name, value in items)


def classify(components: tuple[FlowComponent, ...]) -> tuple[str, float]:
    """Stance from the weighted mean of the parts that could be scored."""
    pairs = [(c.name, c.score) for c in components]
    if not has_stock_part(pairs):
        return "unknown", float("nan")
    score = weighted_score(pairs)
    return stance_from_score(score), score


# --------------------------------------------------------------------------
# Fetching and assembly
# --------------------------------------------------------------------------


def _wait() -> None:
    get_limiter("yfinance").wait()


def _prices(symbol: str) -> pd.DataFrame:
    """Daily Open/High/Low/Close with a timezone-free date index."""
    history = fetch_price_history(symbol, period=PRICE_PERIOD)
    frame = history[[c for c in ("Open", "High", "Low", "Close") if c in history]].astype(float)
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize(None) if index.tz is not None else index
    return frame.sort_index()


def _chains(ticker, today: date) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    _wait()
    expirations = [e for e in (ticker.options or ())
                   if 0 <= (date.fromisoformat(e) - today).days <= OPTIONS_HORIZON_DAYS]
    chains = {}
    for expiration in expirations[:MAX_OPTION_EXPIRATIONS]:
        _wait()
        chain = ticker.option_chain(expiration)
        chains[expiration] = (chain.calls, chain.puts)
    return chains


def _issuance(symbol: str, ticker, info: dict, close, today: date,
              notes: list[str]) -> tuple[FlowComponent, tuple[QuarterBuyback, ...]]:
    """SEC 10-K for US filers (when enabled), else yfinance's annual statements."""
    inputs = None
    if sec_edgar.ENABLED:
        try:
            cik = sec_edgar.cik_for(symbol)
            if cik is not None:
                inputs = sec_issuance_inputs(
                    sec_edgar.latest_annual_report(sec_edgar.company_facts(cik), as_of=today))
        except Exception as exc:  # SEC is a cross-check; its absence is not fatal
            notes.append(f"SEC data unavailable ({type(exc).__name__}: {exc}); used yfinance.")
    quarters: tuple[QuarterBuyback, ...] = ()
    try:
        _wait()
        quarters = quarterly_buybacks(ticker.quarterly_cashflow, close)
        if inputs is None or inputs.net_issuance is None:
            _wait()
            balance_sheet = ticker.balance_sheet
            _wait()
            cashflow = ticker.cashflow
            _wait()
            inputs = replace(yfinance_issuance_inputs(balance_sheet, cashflow, ticker.splits),
                             currency=currency.report_currency(info))
    except Exception as exc:
        if inputs is None:
            inputs = IssuanceInputs(None, None, f"statements unavailable ({type(exc).__name__})",
                                    UNKNOWN)
    # Buybacks are in the statements' currency, so the market cap must be too.
    cap = info.get("marketCap")
    cap = currency.convert(float(cap), currency.quote_currency(info), inputs.currency) if cap \
        else None
    return issuance_component(inputs, cap), quarters


def analyze(symbol: str, today: date | None = None) -> FlowRead:
    """Read the flow conditions for `symbol`. A part that fails is unknown, not fatal."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    today = today or datetime.now(timezone.utc).date()
    ticker = yf.Ticker(symbol)
    notes: list[str] = []

    info: dict = {}
    try:
        _wait()
        info = ticker.info or {}
    except Exception as exc:
        notes.append(f"Company info unavailable ({type(exc).__name__}).")

    close = None
    try:
        prices = _prices(symbol)
        close = prices["Close"]
        stock = stock_volatility_component(prices, symbol)
    except Exception as exc:
        stock = FlowComponent("systematic_stock", "Systematic funds: the stock's volatility",
                              None, f"Price history unavailable ({type(exc).__name__}).")

    try:
        market = market_component(_prices(MARKET_PROXY)["Close"])
    except Exception as exc:
        market = FlowComponent("systematic_market", f"Systematic funds: market "
                               f"({MARKET_PROXY})", None,
                               f"{MARKET_PROXY} history unavailable ({type(exc).__name__}).")

    issuance, quarters = _issuance(symbol, ticker, info, close, today, notes)

    options, options_note = None, ""
    try:
        chains = _chains(ticker, today)
        if close is not None and chains:
            options = options_snapshot(chains, float(close.iloc[-1]),
                                       realised_volatility(close), today)
        elif close is None:
            options_note = "No price to centre the options on (price history unavailable)."
        else:
            options_note = (f"No option expirations within {OPTIONS_HORIZON_DAYS} days were "
                            f"returned (none listed, or the request failed).")
    except Exception as exc:
        options_note = f"Options data unavailable ({type(exc).__name__})."

    components = (issuance, market, stock)
    signal, score = classify(components)
    reasons = [f"{c.label}: {c.reason}" for c in components]
    if signal == "unknown" and market.score is not None:
        reasons.append("No flow data about the stock itself, so no stance: the market's "
                       "setting alone does not speak for one stock.")
    if options is not None:
        iv = (f"; at-the-money implied volatility {options.atm_implied_vol:.0%} "
              f"({options.atm_expiration}) against {options.realised_vol:.0%} realised over the "
              f"last month" if math.isfinite(options.atm_implied_vol) else "")
        reasons.append(
            f"Options (not scored): put/call open interest {options.put_call_ratio:.2f} "
            f"({options.put_open_interest:,} puts, {options.call_open_interest:,} calls, "
            f"expirations to {options.expirations[-1]}){iv}.")
    elif options_note:
        reasons.append(f"Options (not scored): {options_note}")
    institutions = info.get("heldPercentInstitutions")
    insiders = info.get("heldPercentInsiders")
    if institutions is not None or insiders is not None:
        parts = []
        if institutions is not None:
            parts.append(f"institutions hold {float(institutions):.0%}")
        if insiders is not None:
            parts.append(f"insiders {float(insiders):.0%}")
        reasons.append(f"Ownership (not scored): {', '.join(parts)}.")
    return FlowRead(
        symbol=symbol,
        signal=signal,
        score=score,
        components=components,
        reasons=reasons,
        options=options,
        options_note=options_note,
        quarters=quarters,
        institutions_pct=float(institutions) if institutions is not None else None,
        insiders_pct=float(insiders) if insiders is not None else None,
        notes=" ".join(notes),
        report_currency=currency.report_currency(info),
    )
