"""Volatility estimators for the flow analyst's volatility-regime signal.

The signal compares the stock's volatility over the last month with its
volatility over the last year. A 21-day close-to-close estimate is too noisy
for that: its standard error in log terms is about 1 / sqrt(2 x 20) = 16%, so
a stock whose volatility never changes crosses the 0.8 or 1.25 thresholds
about 15% of the time by chance (owner's decision 2026-10-22: fix the
estimator, keep the thresholds).

**Yang-Zhang** (Yang & Zhang 2000, "Drift-independent volatility estimation
based on high, low, open, and close prices", J. Business) uses all four daily
prices. For n days, with o = ln(Open / previous Close), c = ln(Close / Open),
u = ln(High / Open) and d = ln(Low / Open):

    var_overnight = sample variance of o
    var_open_close = sample variance of c
    var_rs = mean of u (u - c) + d (d - c)          (Rogers-Satchell)
    k = (alpha - 1) / (alpha + (n + 1) / (n - 1))   (alpha = 1.34 in the paper)
    var_yz = var_overnight + k var_open_close + (1 - k) var_rs

It is unbiased for continuous Brownian prices with opening jumps and does not
depend on the drift. The long window stays close-to-close over 252 days.

**Data-quality guard.** Yang-Zhang needs real intraday prices. A row whose
High equals its Low, whose Open is missing, or whose Open equals the previous
Close is a sign of a stale or synthesised bar. When such rows exceed a share
of the short window (config), the estimate falls back to close-to-close and
says so (`cc_fallback`).

**Continuous score.** z = ln(short_vol / long_vol) / SE, where SE is the
standard error of that log ratio: sqrt(1 / (2 n_short_eff) + 1 / (2 n_long)),
with n the number of returns minus one and n_short_eff scaled by the
estimator's efficiency relative to close-to-close (config, calibrated by the
intraday simulation in tests/test_volatility_simulation.py). The covariance
between the nested windows is ignored, which overstates SE slightly and so
errs toward smaller scores. SE assumes normal returns; fat tails make the true
error larger.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import load_scoring_config

_CFG = load_scoring_config()["flows"]["stock_volatility"]

YANG_ZHANG_ALPHA = float(_CFG["yang_zhang_alpha"])
YZ_EFFICIENCY = float(_CFG["yz_efficiency"])
MAX_BAD_ROW_SHARE = float(_CFG["quality"]["max_bad_row_share"])

YANG_ZHANG = "yang_zhang"
CC_FALLBACK = "cc_fallback"
CLOSE_TO_CLOSE = "close_to_close"


@dataclass(frozen=True)
class VolatilityRatio:
    short_vol: float  # daily, over the short window
    long_vol: float  # daily, close-to-close over the long window
    ratio: float
    z: float  # ln(ratio) / se
    se: float
    estimator: str  # YANG_ZHANG, CC_FALLBACK or CLOSE_TO_CLOSE
    bad_row_share: float  # share of short-window rows failing the quality guard
    n_short: int
    n_long: int


def close_to_close_vol(close: pd.Series | np.ndarray, window: int) -> float:
    """Sample std of the last `window` daily log returns; NaN when too short."""
    values = np.asarray(close, dtype=float)
    values = values[np.isfinite(values)]
    returns = np.diff(np.log(values))
    if len(returns) < window or window < 2:
        return float("nan")
    return float(returns[-window:].std(ddof=1))


def yang_zhang_vol(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    alpha: float = YANG_ZHANG_ALPHA,
) -> float:
    """Yang-Zhang daily volatility over the last n days.

    The arrays hold n + 1 rows: the first row only supplies the previous close
    for the first overnight return.
    """
    o_, h, l_, c_ = (np.asarray(a, dtype=float) for a in (open_, high, low, close))
    n = len(c_) - 1
    if n < 2:
        return float("nan")
    overnight = np.log(o_[1:] / c_[:-1])
    open_close = np.log(c_[1:] / o_[1:])
    up = np.log(h[1:] / o_[1:])
    down = np.log(l_[1:] / o_[1:])
    if not all(np.isfinite(a).all() for a in (overnight, open_close, up, down)):
        return float("nan")
    var_overnight = overnight.var(ddof=1)
    var_open_close = open_close.var(ddof=1)
    var_rs = float(np.mean(up * (up - open_close) + down * (down - open_close)))
    k = (alpha - 1.0) / (alpha + (n + 1.0) / (n - 1.0))
    variance = var_overnight + k * var_open_close + (1.0 - k) * var_rs
    return float(math.sqrt(variance)) if variance > 0 else float("nan")


def bad_row_share(ohlc: pd.DataFrame) -> float:
    """Share of rows (after the first) with High == Low, no Open, or Open == previous Close."""
    if len(ohlc) < 2:
        return float("nan")
    rows = ohlc.iloc[1:]
    previous_close = ohlc["Close"].shift(1).iloc[1:]
    bad = (
        (rows["High"] == rows["Low"])
        | rows["Open"].isna()
        | (rows["Open"] == previous_close)
    )
    return float(bad.mean())


def log_ratio_se(n_short: int, n_long: int, efficiency: float = 1.0) -> float:
    """Standard error of ln(short_vol / long_vol) for independent windows.

    Var(ln sigma_hat) is about 1 / (2 (n - 1)) for n normal returns; a more
    efficient estimator behaves like a sample `efficiency` times larger.
    """
    return math.sqrt(1.0 / (2.0 * (n_short - 1) * efficiency) + 1.0 / (2.0 * (n_long - 1)))


def volatility_ratio(ohlc: pd.DataFrame, short: int, long: int) -> VolatilityRatio:
    """Short-window volatility (Yang-Zhang, or close-to-close on bad data) over long."""
    long_vol = close_to_close_vol(ohlc["Close"], long)
    window = ohlc.iloc[-(short + 1):]
    has_ohlc = all(col in ohlc for col in ("Open", "High", "Low"))
    share = bad_row_share(window) if has_ohlc else float("nan")
    short_vol, estimator, efficiency = float("nan"), CLOSE_TO_CLOSE, 1.0
    if has_ohlc and len(window) == short + 1 and share <= MAX_BAD_ROW_SHARE:
        short_vol = yang_zhang_vol(window["Open"].to_numpy(), window["High"].to_numpy(),
                                   window["Low"].to_numpy(), window["Close"].to_numpy())
        estimator, efficiency = YANG_ZHANG, YZ_EFFICIENCY
    if not math.isfinite(short_vol):
        short_vol = close_to_close_vol(ohlc["Close"], short)
        estimator, efficiency = (CC_FALLBACK if has_ohlc else CLOSE_TO_CLOSE), 1.0
    ratio = short_vol / long_vol if long_vol and math.isfinite(long_vol) else float("nan")
    se = log_ratio_se(short, long, efficiency)
    z = math.log(ratio) / se if math.isfinite(ratio) and ratio > 0 else float("nan")
    return VolatilityRatio(short_vol, long_vol, ratio, z, se, estimator, share, short, long)
