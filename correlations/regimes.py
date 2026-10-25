"""Correlation in calm and stressed markets, and in the tails.

Diversification is valued for what it does in a sell-off, and that is when
correlations tend to rise. Four views of it:

- **Volatility regimes:** the market proxy's trailing realised volatility
  splits days into calm and volatile, using
  backtests.outcome_distribution.classify_regimes (the split sits at the
  proxy's own 67th percentile, so "volatile" is relative to that market).
- **VIX:** days with the VIX close above VIX_STRESS_LEVEL against the rest. An
  absolute, market-wide fear gauge, comparable across universes.
- **Named stress windows** from config (2008 crisis, March 2020, 2022 rate
  shock). These are short: by agreement with the project owner they may go
  down to STRESS_MIN_OBSERVATIONS (40) instead of the usual 60, and every such
  estimate carries a warning with its standard error.
- **Conditional and tail dependence** per pair: correlation on days both
  assets fell, and on the market's worst decile; and how often two assets
  land in their own worst 5% or 10% together.

The trap in the last group: conditioning changes a correlation even when the
underlying relationship has not changed at all (Boyer, Gibson & Loretan 1999;
Forbes & Rigobon 2002). Selecting days when both assets fell truncates their
distribution, which lowers the measured correlation of a perfectly normal,
constant-correlation pair. So every conditional and tail figure is paired with
its **Gaussian benchmark**: the same statistic on a large simulated sample from
a multivariate normal with the full-sample means and covariance, conditioned
identically.

The Gaussian benchmark is not enough on its own. Measured against it, 392 of
435 pairs in the 30-asset universe looked "more correlated on down days",
because real returns have fat tails and volatility clustering, both of which
raise conditional correlation under any symmetric distribution. The test that
isolates genuine asymmetry compares each pair's correlation on joint-down days
with its correlation on joint-up days (Ang & Chen 2002; Hong, Tu & Zhou 2007):
any elliptical distribution, normal or fat-tailed, gives the same value for
both, so a gap can only come from asymmetric dependence. That difference is
tested per pair (Fisher z, two independent samples) with Benjamini-Hochberg
across pairs; on the same data, 86 pairs correlate at least 0.10 more on the
way down and 1 on the way up.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.risk_manager import MIN_OBSERVATIONS
from backtests.outcome_distribution import classify_regimes
from correlations.estimators import CorrelationEstimate, pearson
from correlations.stats import benjamini_hochberg

# CBOE VIX above 30 is the conventional line for a stressed market: it has
# been crossed in every major sell-off since 1990 and is rarely held for long.
VIX_STRESS_LEVEL = 30.0

# Named stress windows may use as few as 40 observations (the project owner's
# decision, so that March 2020 with ~50 trading days gets a number). Every
# estimate between this and MIN_OBSERVATIONS carries a warning.
STRESS_MIN_OBSERVATIONS = 40

# "The market's worst days": the proxy's bottom decile of daily returns.
MARKET_TAIL_PERCENTILE = 10

# Tail-dependence thresholds: worst 5% and 10% of each asset's own returns.
TAIL_QUANTILES = (0.05, 0.10)

# Simulated draws for the Gaussian benchmarks. At 200,000 the benchmark's own
# sampling error is under 0.01, small next to the observed figures' error.
SIMULATION_DRAWS = 200_000
SIMULATION_SEED = 42

# Block bootstrap for the down-versus-up test: 21-day blocks (a trading month)
# keep volatility clustering inside each resample; 200 resamples put the
# standard error's own error near 5%, plenty for a significance call.
ASYMMETRY_BLOCK = 21
ASYMMETRY_BOOTSTRAPS = 200

# A pair whose stressed correlation exceeds its calm one by at least this much
# is reported as losing diversification. 0.20 is several standard errors for
# the sample sizes involved and large enough to change a portfolio decision.
DIVERSIFICATION_LOSS = 0.20


@dataclass(frozen=True)
class RegimeEstimate:
    name: str
    label: str
    definition: str
    estimate: CorrelationEstimate
    warnings: tuple[str, ...] = ()

    @property
    def n_observations(self) -> int:
        return self.estimate.n_observations


@dataclass(frozen=True)
class ConditionalPair:
    asset_a: str
    asset_b: str
    condition: str
    observed: float
    benchmark: float  # the same statistic under a normal model with the full-sample correlation
    n_observations: int

    @property
    def excess(self) -> float:
        return self.observed - self.benchmark


@dataclass(frozen=True)
class AsymmetryPair:
    """Correlation on joint-down days against joint-up days, with its test."""

    asset_a: str
    asset_b: str
    down: float
    up: float
    n_down: int
    n_up: int
    p_value: float
    p_adjusted: float = float("nan")
    significant_adjusted: bool = False

    @property
    def difference(self) -> float:
        return self.down - self.up


@dataclass(frozen=True)
class TailPair:
    asset_a: str
    asset_b: str
    quantile: float
    lower: float  # share of days in both assets' worst-q tails, divided by q
    lower_benchmark: float
    upper: float
    upper_benchmark: float
    joint_lower_days: int

    @property
    def lower_excess(self) -> float:
        return self.lower - self.lower_benchmark


@dataclass(frozen=True)
class DiversificationShift:
    asset_a: str
    asset_b: str
    calm: float
    stressed: float
    stress_definition: str

    @property
    def change(self) -> float:
        return self.stressed - self.calm


@dataclass(frozen=True)
class RegimeAnalysis:
    regimes: tuple[RegimeEstimate, ...] = ()
    conditional: tuple[ConditionalPair, ...] = ()
    tails: tuple[TailPair, ...] = ()
    shifts: tuple[DiversificationShift, ...] = ()
    notes: tuple[str, ...] = ()
    asymmetry: tuple[AsymmetryPair, ...] = ()

    def regime(self, name: str) -> RegimeEstimate | None:
        return next((r for r in self.regimes if r.name == name), None)


def _subset_estimate(
    name: str,
    label: str,
    definition: str,
    returns: pd.DataFrame,
    mask: pd.Series,
    convention: str,
    min_observations: int = MIN_OBSERVATIONS,
) -> RegimeEstimate:
    subset = returns[mask.reindex(returns.index).fillna(False).astype(bool)]
    estimate = pearson(subset, "daily", convention, min_observations=min_observations)
    return RegimeEstimate(name, label, definition, estimate, estimate.warnings)


def volatility_regimes(
    returns: pd.DataFrame, proxy_returns: pd.Series, proxy: str, convention: str = ""
) -> tuple[RegimeEstimate, ...]:
    labels, threshold = classify_regimes(proxy_returns.dropna())
    definition = (
        f"{proxy} trailing 21-day realised volatility {{}} {threshold:.0f}% annualised "
        f"(its 67th percentile)"
    )
    return (
        _subset_estimate("calm", "Calm", definition.format("at or below"), returns,
                         labels == "calm", convention),
        _subset_estimate("volatile", "Volatile", definition.format("above"), returns,
                         labels == "volatile", convention),
    )


def vix_regimes(
    returns: pd.DataFrame, vix: pd.Series, convention: str = "", level: float = VIX_STRESS_LEVEL
) -> tuple[RegimeEstimate, ...]:
    aligned = vix.reindex(returns.index)
    missing = int(aligned.isna().sum())
    notes = (f"VIX missing on {missing} return date(s); those dates are left out.",) if missing else ()
    high = _subset_estimate(
        "vix_high", f"VIX > {level:.0f}", f"VIX close above {level:.0f} on the same day",
        returns, aligned > level, convention,
    )
    low = _subset_estimate(
        "vix_low", f"VIX <= {level:.0f}", f"VIX close at or below {level:.0f} on the same day",
        returns, aligned <= level, convention,
    )
    return (
        RegimeEstimate(high.name, high.label, high.definition, high.estimate,
                       high.warnings + notes),
        RegimeEstimate(low.name, low.label, low.definition, low.estimate, low.warnings + notes),
    )


def stress_windows(
    returns: pd.DataFrame, windows: Mapping[str, Mapping], convention: str = ""
) -> tuple[RegimeEstimate, ...]:
    out = []
    first, last = returns.index.min(), returns.index.max()
    for name, spec in windows.items():
        start, end = pd.Timestamp(str(spec["start"])), pd.Timestamp(str(spec["end"]))
        label = str(spec.get("label", name))
        definition = f"{start.date()} to {end.date()}"
        if end < first or start > last:
            empty = pearson(returns.iloc[0:0], "daily", convention)
            out.append(
                RegimeEstimate(
                    name, label, definition, empty,
                    (f"Not covered: the common history runs {first.date()} to {last.date()}.",),
                )
            )
            continue
        mask = pd.Series((returns.index >= start) & (returns.index <= end), index=returns.index)
        out.append(
            _subset_estimate(name, label, definition, returns, mask, convention,
                             min_observations=STRESS_MIN_OBSERVATIONS)
        )
    return tuple(out)


def _simulate(returns: pd.DataFrame, proxy_returns: pd.Series | None) -> pd.DataFrame:
    """Draws from a multivariate normal with the sample means and covariance."""
    frame = returns.copy()
    if proxy_returns is not None:
        frame["__proxy__"] = proxy_returns.reindex(frame.index)
        frame = frame.dropna()
    rng = np.random.default_rng(SIMULATION_SEED)
    draws = rng.multivariate_normal(
        frame.mean().to_numpy(), frame.cov().to_numpy(), size=SIMULATION_DRAWS,
        method="eigh",
    )
    return pd.DataFrame(draws, columns=frame.columns)


def _pairwise_masked_corr(x: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of every column pair on rows where both columns' masks hold.

    With D the 0/1 mask matrix and Y = D * X, the masked sums for pair (a, b)
    are entries of D'D (count), Y'D and (D*X^2)'D (first and second moments of
    a on the joint rows) and Y'Y (cross moment). Five matrix products replace
    one masked correlation per pair.
    """
    d = masks.astype(float)
    y = d * x
    count = d.T @ d
    sum_a = y.T @ d  # [a, b]: sum of x_a over rows where both a and b qualify
    sum_sq_a = (y * x).T @ d
    cross = y.T @ y
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a, mean_b = sum_a / count, sum_a.T / count
        var_a = sum_sq_a / count - mean_a**2
        var_b = sum_sq_a.T / count - mean_b**2
        cov = cross / count - mean_a * mean_b
        corr = cov / np.sqrt(var_a * var_b)
    corr[(count < 3) | ~np.isfinite(corr)] = np.nan
    return corr, count


def _common_mask_corr(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Correlation matrix on the rows where one shared mask holds."""
    n = int(mask.sum())
    if n < 3:
        return np.full((x.shape[1], x.shape[1]), np.nan), n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(x[mask], rowvar=False), n


def conditional_correlations(
    returns: pd.DataFrame,
    proxy_returns: pd.Series,
    proxy: str,
    simulated: pd.DataFrame | None = None,
) -> tuple[ConditionalPair, ...]:
    """Pairwise correlation on down days, up days, and the market's tails."""
    frame = returns.copy()
    frame["__proxy__"] = proxy_returns.reindex(frame.index)
    frame = frame.dropna()
    sim = simulated if simulated is not None else _simulate(returns, proxy_returns)
    assets = list(returns.columns)

    x = frame[assets].to_numpy()
    xs = sim[assets].to_numpy()
    real_proxy, sim_proxy = frame["__proxy__"].to_numpy(), sim["__proxy__"].to_numpy()

    def market_mask(values: np.ndarray, low: bool) -> np.ndarray:
        cut = np.percentile(values, MARKET_TAIL_PERCENTILE if low else 100 - MARKET_TAIL_PERCENTILE)
        return values <= cut if low else values >= cut

    worst = f"{proxy}_worst_{MARKET_TAIL_PERCENTILE}pct"
    best = f"{proxy}_best_{MARKET_TAIL_PERCENTILE}pct"
    tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for condition, sign in (("both_down", -1.0), ("both_up", 1.0)):
        observed, count = _pairwise_masked_corr(x, sign * x > 0)
        benchmark, _ = _pairwise_masked_corr(xs, sign * xs > 0)
        tables[condition] = (observed, benchmark, count)
    for condition, low in ((worst, True), (best, False)):
        observed, n = _common_mask_corr(x, market_mask(real_proxy, low))
        benchmark, _ = _common_mask_corr(xs, market_mask(sim_proxy, low))
        tables[condition] = (observed, benchmark, np.full(observed.shape, n))

    out: list[ConditionalPair] = []
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            for condition, (observed, benchmark, count) in tables.items():
                out.append(
                    ConditionalPair(
                        a, assets[j], condition, float(observed[i, j]),
                        float(benchmark[i, j]), int(count[i, j]),
                    )
                )
    return tuple(out)


def _asymmetry_standard_errors(values: np.ndarray) -> np.ndarray:
    """Block-bootstrap standard error of (down-day corr - up-day corr) for every pair."""
    n = len(values)
    rng = np.random.default_rng(SIMULATION_SEED)
    blocks = int(math.ceil(n / ASYMMETRY_BLOCK))
    draws = []
    for _ in range(ASYMMETRY_BOOTSTRAPS):
        starts = rng.integers(0, n - ASYMMETRY_BLOCK + 1, size=blocks)
        index = (starts[:, None] + np.arange(ASYMMETRY_BLOCK)[None, :]).ravel()[:n]
        sample = values[index]
        down, _ = _pairwise_masked_corr(sample, sample < 0)
        up, _ = _pairwise_masked_corr(sample, sample > 0)
        draws.append(down - up)
    return np.nanstd(np.stack(draws), axis=0, ddof=1)


def correlation_asymmetry(
    conditional: tuple[ConditionalPair, ...],
    returns: pd.DataFrame,
) -> tuple[AsymmetryPair, ...]:
    """Down-day minus up-day correlation per pair, bootstrap-tested, BH-adjusted.

    The standard error comes from a block bootstrap of days (see
    ASYMMETRY_BLOCK), not from Fisher's z: Fisher assumes normal data within
    each subsample, and on a symmetric fat-tailed pair it called a chance
    difference of -0.05 significant at p = 3e-6. The bootstrap keeps the fat
    tails and volatility clustering in the error estimate.
    """
    down = {(c.asset_a, c.asset_b): c for c in conditional if c.condition == "both_down"}
    up = {(c.asset_a, c.asset_b): c for c in conditional if c.condition == "both_up"}
    clean = returns.dropna()
    assets = list(clean.columns)
    errors = _asymmetry_standard_errors(clean.to_numpy())
    pairs: list[AsymmetryPair] = []
    for key, d in down.items():
        u = up.get(key)
        if u is None:
            continue
        i, j = assets.index(key[0]), assets.index(key[1])
        se = float(errors[i, j])
        p = float("nan")
        if np.isfinite(d.observed) and np.isfinite(u.observed) and np.isfinite(se) and se > 0:
            z = (d.observed - u.observed) / se
            p = float(math.erfc(abs(z) / math.sqrt(2.0)))
        pairs.append(AsymmetryPair(key[0], key[1], d.observed, u.observed, d.n_observations,
                                   u.n_observations, p))
    adjusted, reject = benjamini_hochberg([a.p_value for a in pairs])
    return tuple(
        AsymmetryPair(a.asset_a, a.asset_b, a.down, a.up, a.n_down, a.n_up, a.p_value,
                      float(p_adj), bool(rej))
        for a, p_adj, rej in zip(pairs, adjusted, reject)
    )


def _pseudo_observations(values: np.ndarray) -> np.ndarray:
    """Ranks scaled into (0, 1): each column's empirical distribution function."""
    ranks = values.argsort(axis=0).argsort(axis=0) + 1.0
    return ranks / (values.shape[0] + 1.0)


def tail_dependence(
    returns: pd.DataFrame, simulated: pd.DataFrame | None = None
) -> tuple[TailPair, ...]:
    """Empirical lower and upper tail dependence at each of TAIL_QUANTILES.

    lambda_L(q) = P(U_a <= q and U_b <= q) / q on rank-transformed data: the
    probability that b is in its worst-q tail given a is (and vice versa). It
    equals q for independent assets and 1 for assets that always crash
    together. The Gaussian benchmark shows what a normal model with the same
    correlation predicts; normal models have no tail dependence in the limit,
    so an observed value well above the benchmark is the fat-tail co-crash
    risk a covariance matrix does not capture.
    """
    clean = returns.dropna()
    sim = simulated if simulated is not None else _simulate(clean, None)
    assets = list(clean.columns)
    u = _pseudo_observations(clean.to_numpy())
    us = _pseudo_observations(sim[assets].to_numpy())
    n, ns = len(u), len(us)

    def joint(values: np.ndarray, mask_fn) -> np.ndarray:
        """Counts of rows where both columns are in the tail: one matrix product."""
        d = mask_fn(values).astype(float)
        return d.T @ d

    counts = {}
    for q in TAIL_QUANTILES:
        counts[q] = (
            joint(u, lambda v: v <= q), joint(u, lambda v: v > 1 - q),
            joint(us, lambda v: v <= q), joint(us, lambda v: v > 1 - q),
        )

    out: list[TailPair] = []
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            for q in TAIL_QUANTILES:
                lo, hi, sim_lo, sim_hi = (c[i, j] for c in counts[q])
                out.append(
                    TailPair(
                        asset_a=a,
                        asset_b=assets[j],
                        quantile=q,
                        lower=float(lo / (n * q)),
                        lower_benchmark=float(sim_lo / (ns * q)),
                        upper=float(hi / (n * q)),
                        upper_benchmark=float(sim_hi / (ns * q)),
                        joint_lower_days=int(round(lo)),
                    )
                )
    return tuple(out)


def diversification_shifts(
    calm: CorrelationEstimate, stressed: CorrelationEstimate, definition: str,
    threshold: float = DIVERSIFICATION_LOSS,
) -> tuple[DiversificationShift, ...]:
    """Pairs whose correlation rises by at least `threshold` from calm to stressed."""
    shifts = []
    assets = list(calm.assets)
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            c, s = calm.pair(a, b), stressed.pair(a, b)
            if np.isnan(c) or np.isnan(s) or s - c < threshold:
                continue
            shifts.append(DiversificationShift(a, b, c, s, definition))
    shifts.sort(key=lambda d: -d.change)
    return tuple(shifts)


def analyze_regimes(
    returns: pd.DataFrame,
    proxy_returns: pd.Series,
    proxy: str,
    vix: pd.Series | None,
    windows: Mapping[str, Mapping],
    convention: str = "",
) -> RegimeAnalysis:
    """Every regime, conditional and tail view for one set of daily returns."""
    notes: list[str] = []
    regimes: list[RegimeEstimate] = list(
        volatility_regimes(returns, proxy_returns, proxy, convention)
    )
    if vix is not None and not vix.empty:
        regimes += vix_regimes(returns, vix, convention)
    else:
        notes.append("VIX unavailable: the VIX regime split was skipped.")
    regimes += stress_windows(returns, windows, convention)

    simulated = _simulate(returns, proxy_returns)
    conditional = conditional_correlations(returns, proxy_returns, proxy, simulated)
    tails = tail_dependence(returns, simulated)

    shifts: tuple[DiversificationShift, ...] = ()
    by_name = {r.name: r for r in regimes}
    calm, high = by_name.get("vix_low"), by_name.get("vix_high")
    if calm is not None and high is not None and high.estimate.is_psd:
        shifts = diversification_shifts(calm.estimate, high.estimate, high.definition)
    return RegimeAnalysis(
        tuple(regimes), conditional, tails, shifts, tuple(notes),
        asymmetry=correlation_asymmetry(conditional, returns),
    )
