"""Non-synchronous closes: lag-1 cross-correlation and daily/weekly divergence.

Paris closes at 17:30 CET, about five and a half hours before New York. News
that lands in the US afternoon moves SPY the same day and SU.PA only the next
morning, so a same-day daily correlation between them misses part of the
co-movement. This is one face of the Epps effect: correlations measured over
short intervals are biased toward zero when the series are not observed at the
same moment.

Two checks make it visible:

- **Lag-1 cross-correlation** for pairs on different continents, in both
  directions. A significant corr(US[t-1], Europe[t]) is the signature of the
  late close spilling into the next European day.
- **Daily vs weekly.** Over a week the timing gap is a small share of the
  interval, so weekly correlations are far less affected. A pair whose daily
  and weekly correlations differ by more than EPPS_FLAG_THRESHOLD is flagged.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.risk_manager import MIN_OBSERVATIONS
from backtests.regression import correlate
from correlations.estimators import CorrelationEstimate, PairStatistic, _with_bh

# Daily and weekly correlations of the same pair routinely differ by a few
# hundredths from sampling noise alone. A gap of 0.15 is beyond that for the
# sample sizes involved (thousands of days, hundreds of weeks), and large
# enough to change a diversification judgement.
EPPS_FLAG_THRESHOLD = 0.15


def cross_continent_pairs(
    assets: list[str], regions: Mapping[str, str]
) -> list[tuple[str, str]]:
    """Pairs whose listings are in different regions."""
    return [
        (a, b)
        for i, a in enumerate(assets)
        for b in assets[i + 1 :]
        if regions.get(a) and regions.get(b) and regions[a] != regions[b]
    ]


def lagged_correlations(
    returns: pd.DataFrame, pairs: list[tuple[str, str]]
) -> tuple[PairStatistic, ...]:
    """Lag-1 cross-correlations in both directions, with BH across the family.

    For a pair (a, b) two statistics are produced: corr(a[t-1], b[t]) noted
    "a leads", and corr(b[t-1], a[t]) noted "b leads".
    """
    clean = returns.dropna(how="any")
    stats: list[PairStatistic] = []
    for a, b in pairs:
        for leader, follower in ((a, b), (b, a)):
            lead = clean[leader].shift(1)
            frame = pd.concat([lead, clean[follower]], axis=1).dropna()
            n = len(frame)
            if n < MIN_OBSERVATIONS:
                stats.append(
                    PairStatistic(
                        a, b, "pearson_lag1", float("nan"), float("nan"), n,
                        note=f"{leader} leads; only {n} observations "
                        f"(minimum {MIN_OBSERVATIONS}).",
                    )
                )
                continue
            result = correlate(frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy())
            stats.append(
                PairStatistic(
                    a, b, "pearson_lag1", result.correlation, result.p_value,
                    result.n_observations, note=f"{leader} leads",
                )
            )
    return _with_bh(stats) if stats else ()


CAUSE_CLOSES = "non-synchronous closes (different continents)"
CAUSE_FX_TIMING = (
    "FX timing: the FX close used for conversion is not taken at the asset's close"
)
CAUSE_OTHER = "not a timing effect: short-horizon dynamics or sampling noise"


@dataclass(frozen=True)
class EppsFlag:
    asset_a: str
    asset_b: str
    daily: float
    weekly: float
    cross_continent: bool
    likely_cause: str = ""

    @property
    def gap(self) -> float:
        return self.weekly - self.daily


def epps_flags(
    daily: CorrelationEstimate,
    weekly: CorrelationEstimate,
    regions: Mapping[str, str],
    threshold: float = EPPS_FLAG_THRESHOLD,
    converted: bool = False,
) -> tuple[EppsFlag, ...]:
    """Pairs whose daily and weekly correlations differ by more than `threshold`.

    Each flag names its likely cause. Cross-continent pairs point to the close
    times. Same-region pairs in a currency-converted view point to the FX
    series, whose daily close is stamped on a different clock from the
    assets'. Same-region pairs in local currency have no timing explanation.
    """
    assets = [a for a in daily.assets if a in weekly.assets]
    flags: list[EppsFlag] = []
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            d, w = daily.pair(a, b), weekly.pair(a, b)
            if np.isnan(d) or np.isnan(w) or abs(w - d) <= threshold:
                continue
            cross = regions.get(a) != regions.get(b)
            cause = CAUSE_CLOSES if cross else CAUSE_FX_TIMING if converted else CAUSE_OTHER
            flags.append(EppsFlag(a, b, d, w, cross_continent=cross, likely_cause=cause))
    flags.sort(key=lambda f: -abs(f.gap))
    return tuple(flags)
