"""Anchored correlation: the recent year, pulled toward its own long run.

Why: correlations mean-revert. Over 39 years of AAPL-MSFT, when the one-year
correlation was 0.3 or lower, the median over the following year was +0.33
(a regression of next year on this year has a slope of about 0.5). A one-year
sample treats today's level as tomorrow's. Ledoit-Wolf does not fix that: its
intensity formula counts only the noise of estimating a *fixed* matrix from
252 days (an intensity of 0.07 on that window), while the variation of the
true correlation from one year to the next is several times larger.

The estimator, at a forecast origin t:

    F = w * A + (1 - w) * B

- A: sample correlation over the last `window` days (the recent year).
- B: sample correlation over all history up to t (the long run).
- w: one weight for the whole matrix, fitted by least squares on earlier
  origins s whose outcome R(s) was already known at t:

      w = sum_s <A_s - B_s, R_s - B_s> / sum_s ||A_s - B_s||^2,  clipped to [0, 1],

  with the inner product over off-diagonal entries and R(s) the correlation
  realised over the `horizon_months` after s. Noise in a realised matrix from
  ~21 days is independent of A and B, so it makes w less precise but does not
  bias it.

One weight, not one per pair: per-pair weights would each be fitted on the
same noisy history and the result need not be a valid matrix, whereas a convex
combination of two correlation matrices always is one (PSD, unit diagonal).
Until MIN_FIT_PERIODS earlier outcomes are known, w is 1 and the estimator is
the plain window sample; the result says so.

All window sums come from prefix sums at the needed boundaries (block matrix
products), so a full pass over 20 years of daily data takes milliseconds and
the walk-forward can refit at every month end.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The recent window: one trading year, as for every structured estimator.
ANCHOR_WINDOW = 252

# Earlier outcomes needed before the weight is fitted rather than fixed at 1.
# Two years of monthly outcomes: fewer and a single regime decides the weight.
MIN_FIT_PERIODS = 24

# A month (or the months of a longer horizon) with fewer trading days per month
# than this gives no realised matrix; only a partial current month is that short.
MIN_DAYS_PER_MONTH = 15


@dataclass(frozen=True)
class AnchorPath:
    """Every forecast origin (month end) with its parts, outcome and weight."""

    dates: pd.DatetimeIndex
    recent: np.ndarray  # K x N x N
    long_run: np.ndarray  # K x N x N
    realised: np.ndarray  # K x N x N; NaN where the outcome is not known yet
    weights: np.ndarray  # K; fitted only on outcomes known at that origin
    n_fit: np.ndarray  # K; how many outcomes the weight rests on
    forecasts: np.ndarray  # K x N x N
    window: int
    horizon_months: int

    def offdiagonal_errors(self, kind: str = "forecast") -> np.ndarray:
        """realised - prediction per origin, off-diagonal entries, fitted origins only."""
        prediction = {"forecast": self.forecasts, "recent": self.recent,
                      "long_run": self.long_run}[kind]
        n = self.recent.shape[1]
        upper = np.triu_indices(n, 1)
        keep = (self.n_fit >= MIN_FIT_PERIODS) & np.isfinite(self.realised).all(axis=(1, 2))
        return (self.realised[keep] - prediction[keep])[:, upper[0], upper[1]]


@dataclass(frozen=True)
class AnchoredCorrelation:
    correlation: np.ndarray
    weight: float
    n_fit: int
    recent: np.ndarray
    long_run: np.ndarray
    long_run_observations: int
    warnings: tuple[str, ...] = ()
    path: AnchorPath | None = field(default=None, repr=False)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last date of each calendar month present in `index` (the last one may be partial)."""
    index = pd.DatetimeIndex(index).sort_values()
    if index.empty:
        return index
    last = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    return pd.DatetimeIndex(last.to_numpy())


def _prefix_sums(values: np.ndarray, boundaries: np.ndarray):
    """n, sum x and sum x x' over rows 0..b for each boundary b (b = -1 gives zeros)."""
    order = np.unique(boundaries)
    n_assets = values.shape[1]
    count = {-1: 0}
    first = {-1: np.zeros(n_assets)}
    second = {-1: np.zeros((n_assets, n_assets))}
    previous = -1
    for b in order:
        if b < 0:
            continue
        block = values[previous + 1 : b + 1]
        count[b] = count[previous] + len(block)
        first[b] = first[previous] + block.sum(axis=0)
        second[b] = second[previous] + block.T @ block
        previous = b
    return count, first, second


def _correlation(n: int, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Sample correlation from window sums; NaN when undefined."""
    k = len(s1)
    if n < 3:
        return np.full((k, k), np.nan)
    covariance = (s2 - np.outer(s1, s1) / n) / (n - 1)
    sd = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if np.any(sd == 0.0):
        return np.full((k, k), np.nan)
    corr = covariance / np.outer(sd, sd)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def anchor_path(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int = ANCHOR_WINDOW,
    horizon_months: int = 1,
) -> AnchorPath:
    """Forecast at every month end with enough history; see the module docstring."""
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if np.isnan(values).any():
        raise ValueError("anchor_path needs complete rows; drop NaN rows first.")
    if horizon_months < 1:
        raise ValueError("horizon_months must be at least 1.")
    ends = dates.get_indexer(month_ends(dates))
    origins = ends[ends + 1 >= window]
    n_assets = values.shape[1]
    if origins.size == 0:
        raise ValueError(f"Need at least {window} observations, got {len(values)}.")

    boundaries = np.concatenate([ends, origins - window, [-1]])
    count, first, second = _prefix_sums(values, boundaries)

    def window_corr(lo: int, hi: int) -> np.ndarray:  # rows lo+1 .. hi
        return _correlation(count[hi] - count[lo], first[hi] - first[lo], second[hi] - second[lo])

    k = len(origins)
    recent = np.empty((k, n_assets, n_assets))
    long_run = np.empty_like(recent)
    realised = np.full_like(recent, np.nan)
    end_position = {int(p): i for i, p in enumerate(ends)}
    for i, p in enumerate(origins):
        recent[i] = window_corr(int(p) - window, int(p))
        long_run[i] = window_corr(-1, int(p))
        j = end_position[int(p)] + horizon_months
        if j < len(ends):
            q = int(ends[j])
            if q - p >= MIN_DAYS_PER_MONTH * horizon_months:
                realised[i] = window_corr(int(p), q)

    # Least-squares weight from outcomes known at each origin. Origin s's
    # outcome is known at origin t when s + horizon <= t (in month ends).
    upper = np.triu_indices(n_assets, 1)
    a = recent[:, upper[0], upper[1]]
    b = long_run[:, upper[0], upper[1]]
    r = realised[:, upper[0], upper[1]]
    usable = np.isfinite(r).all(axis=1) & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    numerator = np.where(usable, ((a - b) * (np.nan_to_num(r) - b)).sum(axis=1), 0.0)
    denominator = np.where(usable, ((a - b) ** 2).sum(axis=1), 0.0)
    cum_num = np.cumsum(numerator)
    cum_den = np.cumsum(denominator)
    cum_n = np.cumsum(usable.astype(int))

    weights = np.ones(k)
    n_fit = np.zeros(k, dtype=int)
    for i in range(k):
        last_known = i - horizon_months
        if last_known < 0:
            continue
        n_fit[i] = int(cum_n[last_known])
        if n_fit[i] >= MIN_FIT_PERIODS and cum_den[last_known] > 0:
            weights[i] = float(np.clip(cum_num[last_known] / cum_den[last_known], 0.0, 1.0))
    forecasts = weights[:, None, None] * recent + (1.0 - weights[:, None, None]) * long_run
    return AnchorPath(
        dates=dates[origins], recent=recent, long_run=long_run, realised=realised,
        weights=weights, n_fit=n_fit, forecasts=forecasts, window=window,
        horizon_months=horizon_months,
    )


def anchored_correlation(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int = ANCHOR_WINDOW,
    horizon_months: int = 1,
) -> AnchoredCorrelation:
    """The anchored forecast at the last row of `values` (whether or not a month end)."""
    path = anchor_path(values, dates, window, horizon_months)
    weight, n_fit = float(path.weights[-1]), int(path.n_fit[-1])
    warnings: list[str] = []
    if n_fit < MIN_FIT_PERIODS:
        warnings.append(
            f"Only {n_fit} earlier outcome(s) known (at least {MIN_FIT_PERIODS} needed to fit "
            f"the weight), so the weight is 1: this is the plain {window}-day sample."
        )
    return AnchoredCorrelation(
        correlation=path.forecasts[-1],
        weight=weight,
        n_fit=n_fit,
        recent=path.recent[-1],
        long_run=path.long_run[-1],
        long_run_observations=int(len(values)),
        warnings=tuple(warnings),
        path=path,
    )
