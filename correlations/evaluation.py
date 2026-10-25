"""Walk-forward evaluation: which correlation estimator forecasts best out of sample.

Every estimator in this package describes the past. An allocation needs a
forecast: the correlation over the period the weights will be held. This
module measures that the only way that counts, out of sample:

1. On the last trading day of each month, every estimator sees only the
   trailing ESTIMATION_WINDOW days (the short-window samples see fewer).
2. Its forecast is scored on the following month, which it has not seen.
3. Each estimator is compared with the benchmark, the 252-day sample
   correlation, month by month.

Two scores, because they answer different questions.

**(a) Forecast error.** The Frobenius distance between the forecast and the
correlation matrix realised over the next month. A realised matrix from ~21
daily returns is itself noisy, but that noise is the same for every estimator:
in expectation it adds the same amount to every squared error, so a ranking on
the mean squared Frobenius error is unbiased even though the levels are
inflated (Patton 2011; Laurent, Rombouts & Violante 2013 show this loss is
robust to a noisy proxy). The ranking therefore uses the squared error; the
tables show its root per pair, which is in correlation units.

**(b) Minimum-variance portfolio.** Build the global minimum-variance portfolio
from each forecast and measure the volatility it actually had over the next
month. This is the use an allocation makes of a covariance matrix, and it
punishes the errors that matter: a spuriously low correlation that an
optimiser leans on. To isolate the correlation, every estimator's matrix is
combined with the same trailing 252-day volatilities, Sigma = D C D.

- *Long-only* (the headline, because it is what the allocation tool will
  hold) is a quadratic programme, solved exactly by a primal active-set method
  (Nocedal & Wright, Numerical Optimization, algorithm 16.3). The plan was
  projected gradient onto the simplex, but its convergence rate is set by the
  covariance's condition number, which reaches 10^3-10^5 here (a short-bond ETF
  next to emerging-market equity): thousands of iterations for an approximate
  answer, where the active-set method stops at the exact one in at most a few
  dozen steps. scipy's SLSQP solves the same problem in the tests, as an
  independent check, together with the KKT conditions.
- *Unconstrained* is the closed form w = Sigma^-1 1 / (1' Sigma^-1 1). It reacts
  most to estimation error, so it separates estimators more sharply.

Weights are held fixed for the month (rebalanced daily to target, ignoring
drift) and portfolio returns are simple returns. Turnover is one-way, between
consecutive targets.

Each comparison with the benchmark carries a Diebold-Mariano test on the
monthly loss differences with a Newey-West variance (estimation windows
overlap by eleven months in twelve, so consecutive losses are correlated and a
plain t-test would overstate significance), then Benjamini-Hochberg across
the estimators. The test statistic is compared with the normal distribution;
with 100+ monthly periods the small-sample t correction changes nothing.

Daily returns only: a month holds about four weekly returns, too few for a
realised matrix. Spearman and Kendall are left out: they measure dependence on
their own scales and cannot feed a covariance (see estimators.COVARIANCE_METHODS).

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from agents.risk_manager import TRADING_DAYS
from correlations.anchoring import month_ends
from correlations.estimators import (
    COVARIANCE_WINDOW,
    ROLLING_WINDOWS,
    estimate_covariance,
)
from correlations.stats import benjamini_hochberg

# Trailing window of every structured estimator and of the shared volatilities.
ESTIMATION_WINDOW = COVARIANCE_WINDOW

BENCHMARK = f"sample_{ESTIMATION_WINDOW}"

# A month with fewer common trading days is not scored: its realised matrix
# would rest on too few returns. Only a partial last month is ever this short.
MIN_EVALUATION_DAYS = 15

# Fewer monthly periods than this and estimators are not ranked: two years is
# the least that spans more than one market mood.
MIN_PERIODS = 24

# Fewer paired periods than this and no Diebold-Mariano p-value is given.
MIN_TEST_PERIODS = 12

# Covariance with a condition number above this is treated as singular: its
# inverse is dominated by rounding error.
SINGULAR_CONDITION = 1e12

# KKT multipliers above -KKT_TOLERANCE (on a covariance scaled to unit average
# variance) count as non-negative.
KKT_TOLERANCE = 1e-10

# The active-set method needs at most a few passes per asset; this only stops
# a degenerate case from looping.
MAX_ACTIVE_SET_ITERATIONS = 500

# Below this effective number of holdings (1 / sum w^2) the long-only
# portfolio is essentially one asset, so its volatility says little about the
# correlation estimate: in US dollars a short-treasury ETF takes ~97%.
CONCENTRATED_HOLDINGS = 2.0

FORECAST_ERROR = "forecast_error"
LONG_ONLY = "min_variance_long_only"
UNCONSTRAINED = "min_variance_unconstrained"
METRICS = (FORECAST_ERROR, LONG_ONLY, UNCONSTRAINED)

METRIC_LABELS = MappingProxyType({
    FORECAST_ERROR: "Forecast error (RMSE per pair vs next month's realised correlation)",
    LONG_ONLY: "Long-only minimum-variance portfolio (realised volatility, annualised)",
    UNCONSTRAINED: "Unconstrained minimum-variance portfolio (realised volatility, annualised)",
})


# --------------------------------------------------------------------------
# Minimum-variance solvers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MinVarianceResult:
    weights: np.ndarray  # NaN when no solution
    long_only: bool
    iterations: int = 0
    note: str = ""

    @property
    def solved(self) -> bool:
        return bool(np.all(np.isfinite(self.weights)))


def _failed(n: int, long_only: bool, note: str) -> MinVarianceResult:
    return MinVarianceResult(np.full(n, np.nan), long_only, 0, note)


def _equality_solution(cov: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum variance on the free assets with weights summing to one.

    Returns the full weight vector (zero off the free set) and nu, the
    multiplier of the budget constraint: Sigma_FF w_F = nu 1.
    """
    x = np.linalg.solve(cov[np.ix_(free, free)], np.ones(int(free.sum())))
    total = float(x.sum())
    weights = np.zeros(len(cov))
    weights[free] = x / total
    return weights, 1.0 / total


def min_variance(covariance: np.ndarray, long_only: bool = True) -> MinVarianceResult:
    """Global minimum-variance weights, fully invested, optionally long-only.

    Unconstrained: w = Sigma^-1 1 / (1' Sigma^-1 1). Long-only: primal
    active-set method. Start from equal weights (feasible, nothing at zero);
    repeatedly solve the equality-constrained problem on the assets not held
    at zero. If that solution has a negative weight, step toward it only until
    the first weight reaches zero, and hold that asset at zero. If it is
    feasible, check the KKT multipliers of the assets held at zero,
    mu_i = (Sigma w)_i - nu: all non-negative means optimal (for a convex
    problem the KKT conditions are sufficient); otherwise release the most
    negative one. Each step lowers the variance, so no set of held assets
    repeats and the method ends at the exact optimum.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n) or n == 0:
        raise ValueError("covariance must be a square matrix.")
    if not np.isfinite(cov).all():
        return _failed(n, long_only, "covariance has undefined entries")
    cov = (cov + cov.T) / 2.0
    scale = float(np.trace(cov)) / n
    if scale <= 0.0:
        return _failed(n, long_only, "covariance has no variance")
    cov = cov / scale  # weights are scale-free; tolerances become relative
    if np.linalg.cond(cov) > SINGULAR_CONDITION:
        return _failed(n, long_only, "covariance is numerically singular")

    everything = np.ones(n, dtype=bool)
    if not long_only:
        weights, _ = _equality_solution(cov, everything)
        return MinVarianceResult(weights, False, 1)

    weights = np.full(n, 1.0 / n)
    held_at_zero = np.zeros(n, dtype=bool)
    for iteration in range(1, MAX_ACTIVE_SET_ITERATIONS + 1):
        free = ~held_at_zero
        target, nu = _equality_solution(cov, free)
        blocking = free & (target < 0.0)
        if not blocking.any():
            weights = target
            if not held_at_zero.any():
                return MinVarianceResult(weights, True, iteration)
            multipliers = cov[held_at_zero] @ weights - nu
            if multipliers.min() >= -KKT_TOLERANCE:
                return MinVarianceResult(weights, True, iteration)
            release = np.flatnonzero(held_at_zero)[int(np.argmin(multipliers))]
            held_at_zero[release] = False
            continue
        # Step toward the target until the first free weight reaches zero.
        steps = weights[blocking] / (weights[blocking] - target[blocking])
        step = float(np.clip(steps.min(), 0.0, 1.0))
        weights = weights + step * (target - weights)
        hits = blocking & (weights <= KKT_TOLERANCE)
        hits[np.flatnonzero(blocking)[int(np.argmin(steps))]] = True
        weights[hits] = 0.0
        weights = weights / weights.sum()
        held_at_zero |= hits
    return _failed(n, True, f"no convergence in {MAX_ACTIVE_SET_ITERATIONS} iterations")


# --------------------------------------------------------------------------
# Significance of a loss difference
# --------------------------------------------------------------------------


def newey_west_lag(n: int) -> int:
    """Bartlett-kernel lag by the usual rule floor(4 (n/100)^(2/9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


def diebold_mariano(
    benchmark_losses: np.ndarray, candidate_losses: np.ndarray, lag: int | None = None
) -> tuple[float, float, int]:
    """(statistic, two-sided p-value, periods) for equal expected loss.

    d_t = benchmark loss - candidate loss, so a positive statistic means the
    candidate did better. The variance of the mean of d is the Newey-West
    long-run variance, which allows for the autocorrelation that overlapping
    estimation windows create.
    """
    a = np.asarray(benchmark_losses, dtype=float)
    b = np.asarray(candidate_losses, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = int(d.size)
    if n < MIN_TEST_PERIODS:
        return float("nan"), float("nan"), n
    lag = newey_west_lag(n) if lag is None else int(lag)
    centred = d - d.mean()
    variance = float(centred @ centred) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * float(centred[k:] @ centred[:-k]) / n
    if variance <= 0.0:  # identical forecasts: nothing to test
        return float("nan"), float("nan"), n
    statistic = float(d.mean() / math.sqrt(variance / n))
    return statistic, math.erfc(abs(statistic) / math.sqrt(2.0)), n


# --------------------------------------------------------------------------
# Contestants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Contestant:
    """One estimator as estimate_covariance will run it."""

    key: str
    label: str
    method: str
    window: int
    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    # True: the estimator sees all history up to the forecast date (the long
    # run), not only the trailing window.
    history: bool = False


def default_contestants(windows: Sequence[int] = ROLLING_WINDOWS["daily"]) -> tuple[Contestant, ...]:
    """The sample at each rolling window, then every structured estimator at 252 days."""
    samples = tuple(
        Contestant(f"sample_{w}", f"Sample, {w} days", "pearson", int(w)) for w in windows
    )
    w = ESTIMATION_WINDOW
    structured = (
        Contestant("ewma", "EWMA (RiskMetrics)", "ewma", w),
        Contestant("ledoit_wolf", "Ledoit-Wolf, constant-correlation target", "ledoit_wolf", w,
                   MappingProxyType({"target": "constant_correlation"})),
        Contestant("ledoit_wolf_identity", "Ledoit-Wolf, identity target", "ledoit_wolf", w,
                   MappingProxyType({"target": "identity"})),
        Contestant("denoised", "RMT denoised", "denoised", w),
        Contestant("pca_factor", "PCA factor model", "pca_factor", w),
        Contestant("group_factor", "Market + group factor model", "group_factor", w),
        Contestant("anchored", "Anchored to the long run", "anchored", w, history=True),
    )
    return samples + structured


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorScore:
    """How one estimator did on one metric, against the benchmark."""

    key: str
    label: str
    metric: str
    n_periods: int
    value: float  # RMSE per pair, or annualised realised volatility
    relative_to_benchmark: float  # value / benchmark value - 1; negative is better
    win_rate: float  # share of paired periods with a lower loss than the benchmark
    statistic: float = float("nan")  # Diebold-Mariano; positive means better
    p_value: float = float("nan")
    p_adjusted: float = float("nan")
    significant_adjusted: bool = False
    rank: int = 0
    turnover: float = float("nan")  # mean one-way turnover per rebalance
    effective_holdings: float = float("nan")  # mean 1 / sum(w^2)
    max_weight: float = float("nan")  # mean largest |weight|
    gross_exposure: float = float("nan")  # mean sum |w|
    is_benchmark: bool = False

    @property
    def better(self) -> bool:
        """Beats the benchmark, significantly after the multiple-comparison correction."""
        return self.significant_adjusted and self.statistic > 0

    @property
    def worse(self) -> bool:
        """Loses to the benchmark, significantly after the correction."""
        return self.significant_adjusted and self.statistic < 0


@dataclass(frozen=True)
class MetricRanking:
    metric: str
    label: str
    scores: tuple[EstimatorScore, ...]  # best first

    @property
    def winner(self) -> EstimatorScore | None:
        return self.scores[0] if self.scores else None

    @property
    def benchmark(self) -> EstimatorScore | None:
        return next((s for s in self.scores if s.is_benchmark), None)

    def score(self, key: str) -> EstimatorScore | None:
        return next((s for s in self.scores if s.key == key), None)


@dataclass(frozen=True)
class WalkForwardResult:
    frequency: str
    currency_convention: str
    estimation_window: int
    benchmark: str
    contestants: tuple[Contestant, ...]
    n_periods: int
    first_forecast: pd.Timestamp | None = None
    last_forecast: pd.Timestamp | None = None
    evaluation_end: pd.Timestamp | None = None
    rankings: tuple[MetricRanking, ...] = ()
    # Per period (indexed by forecast date), columns (metric, estimator):
    # squared Frobenius error, or the month's realised variance (mean squared
    # daily return) of the portfolio.
    losses: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    # Daily out-of-sample portfolio returns, columns (metric, estimator).
    portfolio_returns: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    # Average long-only weight per asset, per estimator.
    average_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    warnings: tuple[str, ...] = ()

    def ranking(self, metric: str) -> MetricRanking | None:
        return next((r for r in self.rankings if r.metric == metric), None)

    @property
    def evaluated(self) -> bool:
        return bool(self.rankings)

    def top_holdings(self, key: str, n: int = 3) -> list[tuple[str, float]]:
        """The largest average long-only weights for one estimator."""
        weights = self.average_weights.get(key) or {}
        return sorted(weights.items(), key=lambda item: -item[1])[:n]

    def concentration_note(self) -> str:
        """A caveat when the long-only test is dominated by one asset, else ""."""
        ranking = self.ranking(LONG_ONLY)
        bench = ranking.benchmark if ranking else None
        if bench is None or not bench.effective_holdings < CONCENTRATED_HOLDINGS:
            return ""
        holdings = ", ".join(f"{name} {w:.0%}" for name, w in self.top_holdings(BENCHMARK, 2))
        return (
            f"The long-only portfolio is concentrated (effective holdings "
            f"{bench.effective_holdings:.1f}; on average {holdings}), so its volatility says "
            f"little about the correlations among the other assets. The forecast error and the "
            f"unconstrained portfolio are the better guides here."
        )


# --------------------------------------------------------------------------
# The walk-forward
# --------------------------------------------------------------------------


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last date of each calendar month present in `index`."""
    return month_ends(index)


def _squared_frobenius(forecast: np.ndarray, realised: np.ndarray) -> float:
    if not (np.isfinite(forecast).all() and np.isfinite(realised).all()):
        return float("nan")
    return float(((forecast - realised) ** 2).sum())


def _turnover(weights: list[np.ndarray]) -> float:
    pairs = [0.5 * np.abs(b - a).sum() for a, b in zip(weights[:-1], weights[1:])
             if np.isfinite(a).all() and np.isfinite(b).all()]
    return float(np.mean(pairs)) if pairs else float("nan")


def _not_evaluated(
    reason: str, contestants: tuple[Contestant, ...], convention: str, frequency: str
) -> WalkForwardResult:
    return WalkForwardResult(
        frequency=frequency, currency_convention=convention,
        estimation_window=ESTIMATION_WINDOW, benchmark=BENCHMARK, contestants=contestants,
        n_periods=0, warnings=(reason,),
    )


def run_walk_forward(
    returns: pd.DataFrame,
    market: pd.Series | None = None,
    groups: Mapping[str, str] | None = None,
    contestants: Sequence[Contestant] | None = None,
    currency_convention: str = "",
    frequency: str = "daily",
) -> WalkForwardResult:
    """Monthly walk-forward of every contestant; see the module docstring.

    `returns` are daily log returns (complete rows are used). `market` is the
    market-factor return series the group factor model needs, and `groups`
    its asset labels; without them that contestant is left out.
    """
    contestants = tuple(contestants or default_contestants())
    warnings: list[str] = []
    if frequency != "daily":
        return _not_evaluated(
            "Walk-forward runs on daily returns only: a month holds too few weekly returns "
            "for a realised matrix.", contestants, currency_convention, frequency,
        )
    if BENCHMARK not in {c.key for c in contestants}:
        raise ValueError(f"The contestants must include the benchmark {BENCHMARK!r}.")
    if market is None or not groups:
        dropped = [c.label for c in contestants if c.method == "group_factor"]
        contestants = tuple(c for c in contestants if c.method != "group_factor")
        if dropped:
            warnings.append("No market factor supplied, so the group factor model was not tested.")

    clean = returns.astype(float).dropna(how="any")
    if market is not None:
        market = market.reindex(clean.index)
        if market.isna().any():
            keep = market.notna()
            warnings.append(f"{int((~keep).sum())} date(s) without a market return left out.")
            clean, market = clean[keep], market[keep]
    longest = max(max(c.window for c in contestants), ESTIMATION_WINDOW)
    index = clean.index
    values = clean.to_numpy()
    assets = list(clean.columns)
    n_assets = len(assets)
    if n_assets < 2:
        return _not_evaluated("Need at least two assets.", contestants, currency_convention,
                              frequency)

    ends = rebalance_dates(index)
    positions = index.get_indexer(ends)
    periods: list[tuple[int, int]] = []
    short_months = 0
    for k in range(len(positions) - 1):
        start, stop = int(positions[k]), int(positions[k + 1])
        if start + 1 < longest:
            continue
        if stop - start < MIN_EVALUATION_DAYS:
            short_months += 1
            continue
        periods.append((start, stop))
    if short_months:
        warnings.append(
            f"{short_months} month(s) with fewer than {MIN_EVALUATION_DAYS} trading days not "
            f"scored (a partial current month)."
        )
    if len(periods) < MIN_PERIODS:
        return _not_evaluated(
            f"Only {len(periods)} monthly period(s) after the first {longest} days of history; "
            f"at least {MIN_PERIODS} are needed to rank estimators.",
            contestants, currency_convention, frequency,
        )

    keys = [c.key for c in contestants]
    loss_rows: list[dict] = []
    portfolio_parts: dict[tuple[str, str], list[pd.Series]] = {
        (m, k): [] for m in (LONG_ONLY, UNCONSTRAINED) for k in keys
    }
    weights_path: dict[tuple[str, str], list[np.ndarray]] = {
        (m, k): [] for m in (LONG_ONLY, UNCONSTRAINED) for k in keys
    }
    failures: dict[str, set[str]] = {}

    for start, stop in periods:
        window = clean.iloc[start + 1 - longest : start + 1]
        evaluation = values[start + 1 : stop + 1]
        eval_dates = index[start + 1 : stop + 1]
        simple = np.expm1(evaluation)
        with np.errstate(invalid="ignore", divide="ignore"):
            realised = np.corrcoef(evaluation, rowvar=False)
        vol = window.iloc[-ESTIMATION_WINDOW:].std(ddof=1).to_numpy()
        row: dict = {"forecast_date": index[start], "evaluation_end": index[stop]}

        for c in contestants:
            params = dict(c.params)
            if c.method == "group_factor":
                params.update(groups=groups, market=market.loc[window.index])
            sample = clean.iloc[: start + 1] if c.history else window
            try:
                estimate = estimate_covariance(sample, c.method, c.window, frequency, **params)
                forecast = estimate.correlation.matrix.to_numpy()
                note = estimate.correlation.warnings[-1] if estimate.correlation.warnings else ""
            except (ValueError, np.linalg.LinAlgError) as exc:
                forecast, note = np.full((n_assets, n_assets), np.nan), str(exc)
            if not np.isfinite(forecast).all():
                failures.setdefault(c.key, set()).add(note or "not estimable")
            row[(FORECAST_ERROR, c.key)] = _squared_frobenius(forecast, realised)

            covariance = forecast * np.outer(vol, vol)
            for metric, long_only in ((LONG_ONLY, True), (UNCONSTRAINED, False)):
                result = (min_variance(covariance, long_only) if np.isfinite(covariance).all()
                          else _failed(n_assets, long_only, "forecast not estimable"))
                weights_path[(metric, c.key)].append(result.weights)
                if not result.solved:
                    failures.setdefault(c.key, set()).add(result.note)
                    row[(metric, c.key)] = float("nan")
                    continue
                daily = simple @ result.weights
                portfolio_parts[(metric, c.key)].append(pd.Series(daily, index=eval_dates))
                row[(metric, c.key)] = float(np.mean(daily**2))
        loss_rows.append(row)

    for key, notes in failures.items():
        label = next(c.label for c in contestants if c.key == key)
        warnings.append(f"{label}: not usable in some periods ({'; '.join(sorted(notes))}).")

    frame = pd.DataFrame(loss_rows).set_index("forecast_date")
    evaluation_end = frame.pop("evaluation_end")
    losses = frame
    losses.columns = pd.MultiIndex.from_tuples(losses.columns, names=["metric", "estimator"])
    portfolio = pd.DataFrame({
        key: pd.concat(parts) for key, parts in portfolio_parts.items() if parts
    })
    if not portfolio.empty:
        portfolio.columns = pd.MultiIndex.from_tuples(portfolio.columns,
                                                      names=["metric", "estimator"])

    rankings = tuple(
        _rank(metric, contestants, losses, portfolio, weights_path, n_assets)
        for metric in METRICS
    )
    average_weights = {
        c.key: MappingProxyType(dict(zip(assets, np.round(np.nanmean(
            np.vstack(weights_path[(LONG_ONLY, c.key)]), axis=0), 6))))
        for c in contestants
        if any(np.isfinite(w).all() for w in weights_path[(LONG_ONLY, c.key)])
    }
    return WalkForwardResult(
        frequency=frequency,
        currency_convention=currency_convention,
        estimation_window=ESTIMATION_WINDOW,
        benchmark=BENCHMARK,
        contestants=contestants,
        n_periods=len(periods),
        first_forecast=losses.index.min(),
        last_forecast=losses.index.max(),
        evaluation_end=evaluation_end.max(),
        rankings=rankings,
        losses=losses,
        portfolio_returns=portfolio,
        average_weights=MappingProxyType(average_weights),
        warnings=tuple(warnings),
    )


def _rank(
    metric: str,
    contestants: tuple[Contestant, ...],
    losses: pd.DataFrame,
    portfolio: pd.DataFrame,
    weights_path: Mapping[tuple[str, str], list[np.ndarray]],
    n_assets: int,
) -> MetricRanking:
    """Score every contestant on one metric and order them, best first."""

    def headline(key: str) -> float:
        if metric == FORECAST_ERROR:
            column = losses[(metric, key)]
            if column.notna().sum() == 0:
                return float("nan")
            return math.sqrt(float(column.mean()) / (n_assets * (n_assets - 1)))
        if (metric, key) not in portfolio:
            return float("nan")
        series = portfolio[(metric, key)].dropna()
        return float(series.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(series) > 1 else float(
            "nan")

    bench_losses = losses[(metric, BENCHMARK)].to_numpy()
    bench_value = headline(BENCHMARK)
    raw: list[dict] = []
    for c in contestants:
        candidate = losses[(metric, c.key)].to_numpy()
        paired = np.isfinite(candidate) & np.isfinite(bench_losses)
        value = headline(c.key)
        entry = dict(
            key=c.key, label=c.label, metric=metric,
            n_periods=int(np.isfinite(candidate).sum()), value=value,
            relative_to_benchmark=value / bench_value - 1.0 if bench_value else float("nan"),
            win_rate=float((candidate[paired] < bench_losses[paired]).mean())
            if paired.any() else float("nan"),
            is_benchmark=c.key == BENCHMARK,
        )
        if c.key != BENCHMARK:
            statistic, p_value, _ = diebold_mariano(bench_losses, candidate)
            entry.update(statistic=statistic, p_value=p_value)
        if metric != FORECAST_ERROR:
            path = [w for w in weights_path[(metric, c.key)] if np.isfinite(w).all()]
            if path:
                stacked = np.vstack(path)
                entry.update(
                    turnover=_turnover(weights_path[(metric, c.key)]),
                    effective_holdings=float(np.mean(1.0 / (stacked**2).sum(axis=1))),
                    max_weight=float(np.mean(np.abs(stacked).max(axis=1))),
                    gross_exposure=float(np.mean(np.abs(stacked).sum(axis=1))),
                )
        raw.append(entry)

    tested = [e for e in raw if not e["is_benchmark"]]
    adjusted, reject = benjamini_hochberg([e.get("p_value", float("nan")) for e in tested])
    for e, p_adj, rejected in zip(tested, adjusted, reject):
        e.update(p_adjusted=float(p_adj), significant_adjusted=bool(rejected))

    ordered = sorted(raw, key=lambda e: (not math.isfinite(e["value"]), e["value"], e["key"]))
    scores = tuple(EstimatorScore(rank=i + 1, **e) for i, e in enumerate(ordered))
    return MetricRanking(metric, METRIC_LABELS[metric], scores)
