"""False-signal rates of the volatility-regime estimators under constant volatility.

Every path has the same true volatility throughout, so any +1 (short/long at or
below the settling ratio) or -1 (at or above the rising ratio) is a false
signal. Daily bars come from simulated intraday paths (78 five-minute steps
plus an overnight gap), so High, Low, Open and Close are realistic. Two return
distributions:

- Gaussian.
- Student-t with 4 degrees of freedom: each day's variance is drawn from the
  inverse-gamma mixing distribution of the t (scaled to mean 1), so daily
  returns are t(4) with unit variance while each intraday path stays Brownian.

Run with `pytest -s` to see the table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents import volatility as vol
from config.settings import load_scoring_config

STEPS_PER_DAY = 78  # five-minute bars in a 6.5-hour session
OVERNIGHT_SHARE = 0.2  # share of daily variance arriving between close and open
DAILY_VOL = 0.01
PATHS = 4000
CHUNK = 250
DF = 4
SEED = 20261022

_CFG = load_scoring_config()["flows"]["stock_volatility"]
SHORT, LONG = int(_CFG["short_window"]), int(_CFG["long_window"])
SETTLING, RISING = float(_CFG["settling_ratio"]), float(_CFG["rising_ratio"])
DAYS = LONG + 1  # closes c_0 .. c_LONG give LONG daily returns


def simulate(distribution: str, paths: int, seed: int) -> dict[str, np.ndarray]:
    """Open, High, Low, Close arrays of shape (paths, DAYS)."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty((paths, DAYS)) for k in ("open", "high", "low", "close")}
    for start in range(0, paths, CHUNK):
        n = min(CHUNK, paths - start)
        if distribution == "gaussian":
            variance = np.ones((n, DAYS))
        else:  # t(DF) as a normal scale mixture, scaled to unit variance
            variance = (DF - 2) / rng.chisquare(DF, size=(n, DAYS))
        day_var = DAILY_VOL**2 * variance
        overnight = rng.standard_normal((n, DAYS)) * np.sqrt(OVERNIGHT_SHARE * day_var)
        steps = rng.standard_normal((n, DAYS, STEPS_PER_DAY)) * np.sqrt(
            (1 - OVERNIGHT_SHARE) * day_var / STEPS_PER_DAY)[:, :, None]
        intraday = np.cumsum(steps, axis=2)  # log price relative to the open
        close = np.empty((n, DAYS))
        log_prev = np.zeros(n)
        log_open = np.empty((n, DAYS))
        for d in range(DAYS):
            log_open[:, d] = log_prev + overnight[:, d]
            close[:, d] = log_open[:, d] + intraday[:, d, -1]
            log_prev = close[:, d]
        high = log_open + np.maximum(intraday.max(axis=2), 0.0)
        low = log_open + np.minimum(intraday.min(axis=2), 0.0)
        for key, value in (("open", log_open), ("high", high), ("low", low), ("close", close)):
            out[key][start:start + n] = 100.0 * np.exp(value)
    return out


def estimates(bars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Short-window vol estimates and the 252-day close-to-close long vol, per path."""
    paths = bars["close"].shape[0]
    log_returns = np.diff(np.log(bars["close"]), axis=1)
    long_vol = log_returns[:, -LONG:].std(axis=1, ddof=1)
    yz = np.array([
        vol.yang_zhang_vol(bars["open"][i, -(SHORT + 1):], bars["high"][i, -(SHORT + 1):],
                           bars["low"][i, -(SHORT + 1):], bars["close"][i, -(SHORT + 1):])
        for i in range(paths)
    ])
    return {
        "long": long_vol,
        f"CC-{SHORT}d": log_returns[:, -SHORT:].std(axis=1, ddof=1),
        "CC-63d": log_returns[:, -63:].std(axis=1, ddof=1),
        f"YZ-{SHORT}d": yz,
    }


@pytest.fixture(scope="module")
def results():
    table = {}
    for i, distribution in enumerate(("gaussian", "student_t4")):
        est = estimates(simulate(distribution, PATHS, SEED + i))
        for name in (f"CC-{SHORT}d", "CC-63d", f"YZ-{SHORT}d"):
            ratio = est[name] / est["long"]
            table[(distribution, name)] = {
                "false_plus": float(np.mean(ratio <= SETTLING)),
                "false_minus": float(np.mean(ratio >= RISING)),
                "log_bias": float(np.mean(np.log(est[name] / DAILY_VOL))),
                "log_sd": float(np.std(np.log(est[name]), ddof=1)),
            }
    lines = [
        "",
        f"False signals under constant volatility ({PATHS} paths each; +1 at ratio <= "
        f"{SETTLING}, -1 at ratio >= {RISING}; long window CC-{LONG}d)",
        f"  {'returns':<11} {'estimator':<9} {'false +1':>9} {'false -1':>9} "
        f"{'ln bias':>8} {'sd ln':>7}",
    ]
    for (distribution, name), row in table.items():
        lines.append(
            f"  {distribution:<11} {name:<9} {row['false_plus']:>9.2%} {row['false_minus']:>9.2%} "
            f"{row['log_bias']:>+8.3f} {row['log_sd']:>7.3f}")
    print("\n".join(lines))
    return table


def test_yang_zhang_false_signals_at_most_2_5_percent_per_side_gaussian(results):
    row = results[("gaussian", f"YZ-{SHORT}d")]
    assert row["false_plus"] <= 0.025
    assert row["false_minus"] <= 0.025


def test_yang_zhang_beats_close_to_close_on_the_same_window(results):
    for distribution in ("gaussian", "student_t4"):
        yz = results[(distribution, f"YZ-{SHORT}d")]
        cc = results[(distribution, f"CC-{SHORT}d")]
        assert yz["false_plus"] + yz["false_minus"] < cc["false_plus"] + cc["false_minus"]


def test_configured_efficiency_matches_the_simulation(results):
    measured = (results[("gaussian", f"CC-{SHORT}d")]["log_sd"]
                / results[("gaussian", f"YZ-{SHORT}d")]["log_sd"]) ** 2
    assert vol.YZ_EFFICIENCY == pytest.approx(measured, rel=0.15), (
        f"Simulated efficiency {measured:.2f}; update yz_efficiency in config/scoring.yaml")


def test_close_to_close_matches_its_textbook_error(results):
    # sd of ln(sigma_hat) for n normal returns is about 1 / sqrt(2 (n - 1)).
    row = results[("gaussian", f"CC-{SHORT}d")]
    assert row["log_sd"] == pytest.approx(1 / math.sqrt(2 * (SHORT - 1)), rel=0.1)
