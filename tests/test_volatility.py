"""Yang-Zhang estimator, the OHLC quality guard and the log-ratio standard error."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents import volatility as vol


def _bars(n: int = 300, seed: int = 0, stale_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                         index=pd.bdate_range("2025-01-01", periods=n))
    if stale_from is not None:  # stale bars: open = previous close, high = low = close
        prev = frame["Close"].shift(1)
        frame.iloc[stale_from:, 0] = prev.iloc[stale_from:]
        frame.iloc[stale_from:, 1] = frame["Close"].iloc[stale_from:]
        frame.iloc[stale_from:, 2] = frame["Close"].iloc[stale_from:]
    return frame


def test_yang_zhang_by_hand():
    o = np.array([0.0, 100.0, 103.0, 101.0])
    h = np.array([0.0, 104.0, 106.0, 103.0])
    l_ = np.array([0.0, 99.0, 100.0, 97.0])
    c = np.array([100.0, 102.0, 101.0, 99.0])
    ov, oc = np.log(o[1:] / c[:-1]), np.log(c[1:] / o[1:])
    u, d = np.log(h[1:] / o[1:]), np.log(l_[1:] / o[1:])
    n, alpha = 3, 1.34
    k = (alpha - 1) / (alpha + (n + 1) / (n - 1))
    expected = math.sqrt(ov.var(ddof=1) + k * oc.var(ddof=1)
                         + (1 - k) * np.mean(u * (u - oc) + d * (d - oc)))
    assert vol.yang_zhang_vol(o, h, l_, c, alpha) == pytest.approx(expected, rel=1e-12)


def test_clean_bars_use_yang_zhang():
    reading = vol.volatility_ratio(_bars(), 21, 252)
    assert reading.estimator == vol.YANG_ZHANG and reading.bad_row_share == 0.0
    assert reading.se == pytest.approx(vol.log_ratio_se(21, 252, vol.YZ_EFFICIENCY))


def test_stale_bars_fall_back_to_close_to_close():
    bars = _bars(stale_from=290)  # the last 10 of 21 bars are stale: 48% > 20%
    reading = vol.volatility_ratio(bars, 21, 252)
    assert reading.estimator == vol.CC_FALLBACK
    assert reading.bad_row_share > vol.MAX_BAD_ROW_SHARE
    assert reading.short_vol == pytest.approx(vol.close_to_close_vol(bars["Close"], 21))
    assert reading.se == pytest.approx(vol.log_ratio_se(21, 252, 1.0))


def test_a_few_stale_bars_are_tolerated():
    reading = vol.volatility_ratio(_bars(stale_from=297), 21, 252)  # 3 of 21 = 14%
    assert reading.estimator == vol.YANG_ZHANG


def test_close_only_is_close_to_close():
    reading = vol.volatility_ratio(_bars()[["Close"]], 21, 252)
    assert reading.estimator == vol.CLOSE_TO_CLOSE


def test_log_ratio_se_formula():
    assert vol.log_ratio_se(21, 252) == pytest.approx(math.sqrt(1 / 40 + 1 / 502))
    assert vol.log_ratio_se(21, 252, 4.0) < vol.log_ratio_se(21, 252, 1.0)


def test_short_history_is_nan():
    reading = vol.volatility_ratio(_bars(n=100), 21, 252)
    assert math.isnan(reading.ratio) and math.isnan(reading.z)
