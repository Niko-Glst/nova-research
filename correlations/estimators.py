"""Correlation estimators with one shared interface.

Public API (stable, for the allocation tool and anything else downstream)::

    from correlations import estimate_correlation, CorrelationEstimate

    estimate = estimate_correlation(returns, method="pearson")
    estimate.matrix          # DataFrame, unit diagonal (NaN when not estimable)
    estimate.covariance      # DataFrame or None, same units as `returns`
    estimate.warnings        # why a result is missing or should be distrusted
    estimate.is_psd          # checked on every result

    risk = estimate_covariance(returns)   # for an allocation: the forecast that
    risk.covariance          # won out of sample, annualised, from the last 252 rows
    risk.volatility          # annualised, per asset
    risk.correlation         # the CorrelationEstimate behind it

`estimate_correlation` describes a sample. `estimate_covariance` is the
forecast an optimiser should use: by default the method that won the
walk-forward evaluation (correlations/evaluation.py), on the window it was
tested with.

`returns` is a DataFrame of log returns, one column per asset, as produced by
correlations.data.log_returns. Rows with any NaN are dropped first so that
every pair is measured on the same dates: pairwise-complete correlations look
more efficient but can produce a matrix that is not positive semi-definite.

Methods in phase 1:

- ``pearson``  sample Pearson, the baseline. Linear dependence only.
- ``spearman`` Pearson on ranks. Captures any monotonic relationship and is
  robust to a few extreme days.
- ``kendall``  tau-b. The most robust of the three to outliers; on a bivariate
  normal it relates to Pearson by r = sin(pi * tau / 2).
- ``ewma``     RiskMetrics exponentially weighted correlation, which reacts to
  recent data at the cost of a much smaller effective sample.
- ``rolling``  sample Pearson on the most recent `window` observations.

The sample estimators also carry pairwise significance: each Pearson pair goes
through backtests.regression.correlate (t-test), Spearman through the same test
on ranks, Kendall through its own normal approximation. Benjamini-Hochberg is
applied within each method's family of pairs. EWMA and rolling estimates carry
no p-values: they are not plain sample correlations, so the t-test does not
apply to them.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from agents.risk_manager import MIN_OBSERVATIONS
from backtests.regression import SIGNIFICANCE_LEVEL, correlate
from correlations.factors import (
    PCA_VARIANCE_TARGET,
    market_group_factor_model,
    pca_factor_model,
)
from correlations.anchoring import ANCHOR_WINDOW, anchored_correlation
from correlations.rmt import denoise
from correlations.shrinkage import ledoit_wolf
from correlations.stats import (
    SCIPY_FIX,
    benjamini_hochberg,
    check_psd,
    kendall_backend,
    kendall_matrix,
    scipy_status,
)

# RiskMetrics (J.P. Morgan, 1996) decay factors. 0.94 was fitted for daily
# returns and 0.97 for monthly; 0.97 is the usual choice for weekly data too.
EWMA_LAMBDA = MappingProxyType({"daily": 0.94, "weekly": 0.97})

# Rolling windows. Daily: roughly a quarter, half a year and a year. Weekly
# windows start at 60 because shorter ones fall under MIN_OBSERVATIONS.
ROLLING_WINDOWS = MappingProxyType({"daily": (60, 120, 252), "weekly": (60, 104)})

SAMPLE_METHODS = ("pearson", "spearman", "kendall")
STRUCTURED_METHODS = ("ledoit_wolf", "denoised", "pca_factor", "group_factor")
METHODS = SAMPLE_METHODS + ("ewma", "rolling") + STRUCTURED_METHODS + ("anchored",)

# Methods that use the whole history passed to them, not just the last window.
HISTORY_METHODS = ("anchored",)


@dataclass(frozen=True)
class PairStatistic:
    """One pairwise correlation with its significance, before and after BH."""

    asset_a: str
    asset_b: str
    method: str
    correlation: float
    p_value: float
    n_observations: int
    p_adjusted: float = float("nan")
    significant: bool = False  # raw p < SIGNIFICANCE_LEVEL
    significant_adjusted: bool = False  # Benjamini-Hochberg adjusted p <= level
    note: str = ""


@dataclass(frozen=True)
class CorrelationEstimate:
    """A correlation matrix with everything needed to judge it.

    Frozen, but `matrix` and `covariance` are DataFrames and therefore mutable
    in place: treat them as read-only, and copy before modifying.
    """

    method: str
    matrix: pd.DataFrame
    n_observations: int
    frequency: str = "daily"
    covariance: pd.DataFrame | None = None
    parameters: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    warnings: tuple[str, ...] = ()
    is_psd: bool = False
    min_eigenvalue: float = float("nan")
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    currency_convention: str = ""
    pairs: tuple[PairStatistic, ...] = ()

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.matrix.columns)

    def pair(self, asset_a: str, asset_b: str) -> float:
        """Correlation between two assets (order does not matter)."""
        return float(self.matrix.loc[asset_a, asset_b])


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _prepare(returns: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Complete-case rows only, with a note on how many were dropped."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame with one column per asset.")
    notes: list[str] = []
    clean = returns.astype(float).dropna(how="any")
    dropped = len(returns) - len(clean)
    if dropped:
        notes.append(
            f"{dropped} row(s) with a missing return were dropped so every pair uses the "
            f"same dates."
        )
    return clean, notes


def _not_estimable(
    method: str,
    returns: pd.DataFrame,
    frequency: str,
    reason: str,
    parameters: dict,
    currency_convention: str,
    notes: list[str],
) -> CorrelationEstimate:
    columns = list(returns.columns)
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
    return CorrelationEstimate(
        method=method,
        matrix=matrix,
        n_observations=int(len(returns)),
        frequency=frequency,
        parameters=MappingProxyType(dict(parameters)),
        warnings=tuple(notes + [reason]),
        is_psd=False,
        start=returns.index.min() if len(returns) else None,
        end=returns.index.max() if len(returns) else None,
        currency_convention=currency_convention,
    )


def _constant_columns(returns: pd.DataFrame) -> list[str]:
    return [c for c in returns.columns if float(returns[c].std(ddof=1)) == 0.0]


def _finish(
    method: str,
    matrix: np.ndarray,
    returns: pd.DataFrame,
    frequency: str,
    parameters: dict,
    notes: list[str],
    currency_convention: str,
    covariance: np.ndarray | None = None,
    pairs: tuple[PairStatistic, ...] = (),
    n_observations: int | None = None,
) -> CorrelationEstimate:
    """Wrap a matrix in a CorrelationEstimate, running the validity checks."""
    columns = list(returns.columns)
    constant = _constant_columns(returns)
    matrix = np.array(matrix, dtype=float)
    for name in constant:
        i = columns.index(name)
        matrix[i, :] = np.nan
        matrix[:, i] = np.nan
        notes.append(f"{name} has zero variance in this sample; its correlations are undefined.")

    check = check_psd(matrix)
    warnings = notes + [w for w in check.warnings if not constant]
    frame = pd.DataFrame(matrix, index=columns, columns=columns)
    cov_frame = (
        pd.DataFrame(covariance, index=columns, columns=columns) if covariance is not None else None
    )
    return CorrelationEstimate(
        method=method,
        matrix=frame,
        n_observations=int(n_observations if n_observations is not None else len(returns)),
        frequency=frequency,
        covariance=cov_frame,
        parameters=MappingProxyType(dict(parameters)),
        warnings=tuple(warnings),
        is_psd=check.is_psd,
        min_eigenvalue=check.min_eigenvalue,
        start=returns.index.min() if len(returns) else None,
        end=returns.index.max() if len(returns) else None,
        currency_convention=currency_convention,
        pairs=pairs,
    )


def _with_bh(pairs: list[PairStatistic]) -> tuple[PairStatistic, ...]:
    adjusted, reject = benjamini_hochberg([p.p_value for p in pairs])
    out = []
    for pair, p_adj, rejected in zip(pairs, adjusted, reject):
        out.append(
            PairStatistic(
                asset_a=pair.asset_a,
                asset_b=pair.asset_b,
                method=pair.method,
                correlation=pair.correlation,
                p_value=pair.p_value,
                n_observations=pair.n_observations,
                p_adjusted=float(p_adj),
                significant=bool(pair.p_value < SIGNIFICANCE_LEVEL),
                significant_adjusted=bool(rejected),
                note=pair.note,
            )
        )
    return tuple(out)


def _pairs_from_correlate(returns: pd.DataFrame, method: str) -> tuple[PairStatistic, ...]:
    """Pairwise significance via backtests.regression.correlate."""
    columns = list(returns.columns)
    pairs: list[PairStatistic] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            result = correlate(returns[a].to_numpy(), returns[b].to_numpy(), name=f"{a}|{b}")
            pairs.append(
                PairStatistic(a, b, method, result.correlation, result.p_value,
                              result.n_observations)
            )
    return _with_bh(pairs)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def pearson(
    returns: pd.DataFrame,
    frequency: str = "daily",
    currency_convention: str = "",
    min_observations: int = MIN_OBSERVATIONS,
    with_pairs: bool = True,
) -> CorrelationEstimate:
    """Sample Pearson correlation and covariance (ddof=1).

    `min_observations` may be lowered only for named stress windows (see
    correlations/regimes.py); any estimate below MIN_OBSERVATIONS says so.
    `with_pairs=False` skips the per-pair significance tests, for callers that
    only need the matrix (the walk-forward evaluation fits it thousands of
    times).
    """
    clean, notes = _prepare(returns)
    params: dict = {}
    if len(clean) < min_observations:
        return _not_estimable(
            "pearson", clean, frequency,
            f"Only {len(clean)} observations; at least {min_observations} are needed.",
            params, currency_convention, notes,
        )
    if len(clean) < MIN_OBSERVATIONS:
        notes.append(
            f"Only {len(clean)} observations, below the usual minimum of {MIN_OBSERVATIONS}: "
            f"each correlation here has a standard error of roughly "
            f"{1 / np.sqrt(len(clean)):.2f}."
        )
    values = clean.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(values, rowvar=False)
    covariance = np.cov(values, rowvar=False, ddof=1)
    return _finish(
        "pearson", matrix, clean, frequency, params, notes, currency_convention,
        covariance=covariance,
        pairs=_pairs_from_correlate(clean, "pearson") if with_pairs else (),
    )


def spearman(
    returns: pd.DataFrame, frequency: str = "daily", currency_convention: str = ""
) -> CorrelationEstimate:
    """Spearman rank correlation: Pearson on average ranks (ties share a rank)."""
    clean, notes = _prepare(returns)
    params = {"ties": "average rank"}
    if len(clean) < MIN_OBSERVATIONS:
        return _not_estimable(
            "spearman", clean, frequency,
            f"Only {len(clean)} observations; at least {MIN_OBSERVATIONS} are needed.",
            params, currency_convention, notes,
        )
    ranks = clean.rank(method="average")
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(ranks.to_numpy(), rowvar=False)
    pairs = _pairs_from_correlate(ranks, "spearman")
    return _finish(
        "spearman", matrix, clean, frequency, params, notes, currency_convention, pairs=pairs
    )


def kendall(
    returns: pd.DataFrame, frequency: str = "daily", currency_convention: str = ""
) -> CorrelationEstimate:
    """Kendall tau-b, with tie correction and asymptotic p-values.

    Uses scipy when it loads and the built-in implementation otherwise (see
    correlations/stats.py); the result names the backend, and says how to fix
    a missing scipy, because the built-in one is far slower on long samples.
    """
    clean, notes = _prepare(returns)
    params = {"variant": "tau-b", "backend": kendall_backend()}
    if len(clean) < MIN_OBSERVATIONS:
        return _not_estimable(
            "kendall", clean, frequency,
            f"Only {len(clean)} observations; at least {MIN_OBSERVATIONS} are needed.",
            params, currency_convention, notes,
        )
    available, reason = scipy_status()
    if not available:
        notes.append(
            f"scipy could not be loaded ({reason}), so the slower built-in Kendall ran. "
            f"To fix, run in nova-research: {SCIPY_FIX}"
        )
    tau, p_values = kendall_matrix(clean.to_numpy())
    columns = list(clean.columns)
    pairs = [
        PairStatistic(a, b, "kendall", float(tau[i, j]), float(p_values[i, j]), len(clean))
        for i, a in enumerate(columns)
        for j, b in enumerate(columns)
        if j > i
    ]
    return _finish(
        "kendall", tau, clean, frequency, params, notes, currency_convention,
        pairs=_with_bh(pairs),
    )


def ewma_weights(n: int, lam: float) -> np.ndarray:
    """Normalised RiskMetrics weights, oldest first: w_t proportional to lam^(n-1-t)."""
    raw = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return raw / raw.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2."""
    return float(weights.sum() ** 2 / (weights**2).sum())


def ewma(
    returns: pd.DataFrame,
    frequency: str = "daily",
    lam: float | None = None,
    currency_convention: str = "",
) -> CorrelationEstimate:
    """RiskMetrics EWMA correlation at the last date of `returns`.

    Uses the explicit normalised weighted sum rather than the recursion
    S_t = lam * S_(t-1) + (1 - lam) r_t r_t'. They are the same estimator, but
    the recursion needs a starting matrix, and that arbitrary seed leaks into
    the result when the history is short. Returns are not demeaned, as in
    RiskMetrics: at daily horizons the mean is noise next to the volatility.
    """
    clean, notes = _prepare(returns)
    lam = float(lam if lam is not None else EWMA_LAMBDA.get(frequency, EWMA_LAMBDA["daily"]))
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}.")
    params: dict = {"lambda": lam, "demeaned": False}
    if len(clean) < MIN_OBSERVATIONS:
        return _not_estimable(
            "ewma", clean, frequency,
            f"Only {len(clean)} observations; at least {MIN_OBSERVATIONS} are needed.",
            params, currency_convention, notes,
        )

    values = clean.to_numpy()
    weights = ewma_weights(len(values), lam)
    covariance = (values * weights[:, None]).T @ values
    scale = np.sqrt(np.diag(covariance))
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = covariance / np.outer(scale, scale)

    ess = effective_sample_size(weights)
    params["effective_sample_size"] = round(ess, 1)
    params["half_life"] = round(float(np.log(0.5) / np.log(lam)), 1)
    if ess < MIN_OBSERVATIONS:
        notes.append(
            f"Effective sample size is about {ess:.0f} observations (lambda {lam}), below "
            f"{MIN_OBSERVATIONS}: this estimate reacts quickly but is noisy. That is the "
            f"design trade-off of EWMA, not an error."
        )
    return _finish(
        "ewma", matrix, clean, frequency, params, notes, currency_convention,
        covariance=covariance, n_observations=len(clean),
    )


def rolling_latest(
    returns: pd.DataFrame,
    frequency: str = "daily",
    window: int = 252,
    currency_convention: str = "",
) -> CorrelationEstimate:
    """Sample Pearson on the most recent `window` observations."""
    clean, notes = _prepare(returns)
    params = {"window": int(window)}
    if window < MIN_OBSERVATIONS:
        return _not_estimable(
            "rolling", clean, frequency,
            f"Window {window} is below the {MIN_OBSERVATIONS}-observation minimum.",
            params, currency_convention, notes,
        )
    if len(clean) < window:
        return _not_estimable(
            "rolling", clean, frequency,
            f"Only {len(clean)} observations for a {window}-observation window.",
            params, currency_convention, notes,
        )
    tail = clean.iloc[-window:]
    values = tail.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(values, rowvar=False)
    covariance = np.cov(values, rowvar=False, ddof=1)
    return _finish(
        "rolling", matrix, tail, frequency, params, notes, currency_convention,
        covariance=covariance,
    )


def _too_short(method, clean, frequency, params, convention, notes):
    return _not_estimable(
        method, clean, frequency,
        f"Only {len(clean)} observations; at least {MIN_OBSERVATIONS} are needed.",
        params, convention, notes,
    )


def ledoit_wolf_estimate(
    returns: pd.DataFrame,
    frequency: str = "daily",
    target: str = "constant_correlation",
    currency_convention: str = "",
) -> CorrelationEstimate:
    """Ledoit-Wolf shrunk covariance and its correlation (see correlations/shrinkage.py)."""
    clean, notes = _prepare(returns)
    params: dict = {"target": target}
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("ledoit_wolf", clean, frequency, params, currency_convention, notes)
    constant = _constant_columns(clean)
    if constant:
        return _not_estimable(
            "ledoit_wolf", clean, frequency,
            f"Zero variance in {', '.join(constant)}: shrinkage is undefined.",
            params, currency_convention, notes,
        )
    result = ledoit_wolf(clean.to_numpy(), target)
    params["intensity"] = round(result.intensity, 4)
    if target == "constant_correlation":
        params["average_correlation"] = round(result.average_correlation, 4)
    scale = np.sqrt(np.diag(result.covariance))
    matrix = result.covariance / np.outer(scale, scale)
    np.fill_diagonal(matrix, 1.0)
    return _finish(
        "ledoit_wolf", matrix, clean, frequency, params, notes, currency_convention,
        covariance=result.covariance,
    )


def denoised(
    returns: pd.DataFrame, frequency: str = "daily", currency_convention: str = ""
) -> CorrelationEstimate:
    """Random-matrix-theory denoised correlation (see correlations/rmt.py)."""
    clean, notes = _prepare(returns)
    params: dict = {"method": "constant residual eigenvalue", "edge": "Tracy-Widom 95%"}
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("denoised", clean, frequency, params, currency_convention, notes)
    if _constant_columns(clean):
        return _not_estimable(
            "denoised", clean, frequency, "A zero-variance column makes the sample matrix "
            "undefined.", params, currency_convention, notes,
        )
    sample = np.corrcoef(clean.to_numpy(), rowvar=False)
    try:
        result = denoise(sample, len(clean))
    except ValueError as exc:
        return _not_estimable(
            "denoised", clean, frequency, str(exc), params, currency_convention, notes
        )
    params.update(
        sigma2=round(result.sigma2, 4),
        lambda_plus=round(result.lambda_plus, 4),
        signal_edge=round(result.signal_edge, 4),
        n_signal=result.n_signal,
        q=round(result.q, 1),
    )
    std = clean.std(ddof=1).to_numpy()
    return _finish(
        "denoised", result.correlation, clean, frequency, params,
        notes + list(result.warnings), currency_convention,
        covariance=result.correlation * np.outer(std, std),
    )


def pca_factor(
    returns: pd.DataFrame,
    frequency: str = "daily",
    variance_target: float = PCA_VARIANCE_TARGET,
    currency_convention: str = "",
) -> CorrelationEstimate:
    """Statistical (PCA) factor-model correlation (see correlations/factors.py)."""
    clean, notes = _prepare(returns)
    params: dict = {"variance_target": variance_target}
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("pca_factor", clean, frequency, params, currency_convention, notes)
    if _constant_columns(clean):
        return _not_estimable(
            "pca_factor", clean, frequency, "A zero-variance column makes the model undefined.",
            params, currency_convention, notes,
        )
    result = pca_factor_model(clean.to_numpy(), variance_target)
    params.update(n_factors=result.n_factors, explained=round(result.explained, 4))
    return _finish(
        "pca_factor", result.correlation, clean, frequency, params,
        notes + list(result.warnings), currency_convention, covariance=result.covariance,
    )


def group_factor(
    returns: pd.DataFrame,
    groups: Mapping[str, str],
    market: pd.Series,
    frequency: str = "daily",
    currency_convention: str = "",
    market_name: str = "market",
) -> CorrelationEstimate:
    """Market + group factor-model correlation (see correlations/factors.py)."""
    clean, notes = _prepare(returns)
    params: dict = {"market": market_name}
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("group_factor", clean, frequency, params, currency_convention, notes)
    if _constant_columns(clean):
        return _not_estimable(
            "group_factor", clean, frequency, "A zero-variance column makes the model undefined.",
            params, currency_convention, notes,
        )
    aligned_market = market.reindex(clean.index)
    missing = int(aligned_market.isna().sum())
    if missing:
        notes.append(f"Market factor missing on {missing} date(s); those dates are left out.")
        clean = clean[aligned_market.notna()]
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("group_factor", clean, frequency, params, currency_convention, notes)
    result = market_group_factor_model(clean, groups, aligned_market.dropna())
    params.update(
        factors=", ".join(result.factor_names), explained=round(result.explained, 4)
    )
    return _finish(
        "group_factor", result.correlation, clean, frequency, params,
        notes + list(result.warnings), currency_convention, covariance=result.covariance,
    )


def anchored(
    returns: pd.DataFrame,
    frequency: str = "daily",
    window: int = ANCHOR_WINDOW,
    horizon_months: int = 1,
    currency_convention: str = "",
) -> CorrelationEstimate:
    """The last `window` days pulled toward the long run (see correlations/anchoring.py).

    Uses every row of `returns`: the recent part is the last `window` rows and
    the long run all of them, so pass the full history. The covariance uses
    the volatilities of the last `window` rows, like every other estimator.
    """
    clean, notes = _prepare(returns)
    params: dict = {"window": int(window), "horizon_months": int(horizon_months)}
    if frequency != "daily":
        return _not_estimable(
            "anchored", clean, frequency, "Daily returns only: the weight is fitted on monthly "
            "outcomes, which a handful of weekly returns cannot measure.",
            params, currency_convention, notes,
        )
    if len(clean) < MIN_OBSERVATIONS:
        return _too_short("anchored", clean, frequency, params, currency_convention, notes)
    if _constant_columns(clean):
        return _not_estimable(
            "anchored", clean, frequency, "A zero-variance column makes the sample undefined.",
            params, currency_convention, notes,
        )
    window = min(int(window), len(clean))
    result = anchored_correlation(clean.to_numpy(), clean.index, window, horizon_months)
    params.update(
        weight=round(result.weight, 4),
        n_fit=result.n_fit,
        long_run_start=str(clean.index[0].date()),
    )
    std = clean.iloc[-window:].std(ddof=1).to_numpy()
    return _finish(
        "anchored", result.correlation, clean, frequency, params,
        notes + list(result.warnings), currency_convention,
        covariance=result.correlation * np.outer(std, std),
    )


# --------------------------------------------------------------------------
# Rolling paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairPath:
    """How one pair's rolling correlation moved over the sample."""

    asset_a: str
    asset_b: str
    minimum: float
    maximum: float
    std: float
    latest: float
    full_sample: float

    @property
    def range(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class RollingCorrelation:
    """Rolling correlation for every pair at one window length."""

    window: int
    frequency: str
    series: pd.DataFrame = field(repr=False)  # columns "A|B", indexed by window end date
    paths: tuple[PairPath, ...] = ()
    latest: CorrelationEstimate | None = None
    warnings: tuple[str, ...] = ()


def rolling_paths(
    returns: pd.DataFrame,
    window: int,
    frequency: str = "daily",
    currency_convention: str = "",
) -> RollingCorrelation:
    """Rolling Pearson correlation for every pair, plus its range and spread.

    The spread of a pair's rolling correlation is the plainest measure of the
    instability a single full-sample number hides.
    """
    clean, notes = _prepare(returns)
    if window < MIN_OBSERVATIONS or len(clean) < window:
        reason = (
            f"Window {window} is below the {MIN_OBSERVATIONS}-observation minimum."
            if window < MIN_OBSERVATIONS
            else f"Only {len(clean)} observations for a {window}-observation window."
        )
        return RollingCorrelation(
            window, frequency, pd.DataFrame(index=clean.index), warnings=tuple(notes + [reason])
        )

    columns = list(clean.columns)
    full = clean.corr()
    series: dict[str, pd.Series] = {}
    paths: list[PairPath] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            path = clean[a].rolling(window).corr(clean[b]).dropna()
            series[f"{a}|{b}"] = path
            if path.empty:
                continue
            paths.append(
                PairPath(
                    asset_a=a,
                    asset_b=b,
                    minimum=float(path.min()),
                    maximum=float(path.max()),
                    std=float(path.std(ddof=1)) if len(path) > 1 else float("nan"),
                    latest=float(path.iloc[-1]),
                    full_sample=float(full.loc[a, b]),
                )
            )
    return RollingCorrelation(
        window=window,
        frequency=frequency,
        series=pd.DataFrame(series),
        paths=tuple(paths),
        latest=rolling_latest(clean, frequency, window, currency_convention),
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


def estimate_correlation(
    returns: pd.DataFrame,
    method: str = "pearson",
    frequency: str = "daily",
    currency_convention: str = "",
    **params,
) -> CorrelationEstimate:
    """Estimate a correlation matrix with the named method.

    Extra keyword arguments go to the estimator: ``lam`` for ``ewma``,
    ``window`` for ``rolling``, ``target`` for ``ledoit_wolf``,
    ``variance_target`` for ``pca_factor``, and ``groups`` plus ``market``
    (a return Series) for ``group_factor``.
    """
    method = method.strip().lower()
    if method == "pearson":
        return pearson(returns, frequency, currency_convention, **params)
    if method == "spearman":
        return spearman(returns, frequency, currency_convention, **params)
    if method == "kendall":
        return kendall(returns, frequency, currency_convention, **params)
    if method == "ewma":
        return ewma(returns, frequency, currency_convention=currency_convention, **params)
    if method == "rolling":
        return rolling_latest(returns, frequency, currency_convention=currency_convention, **params)
    if method == "ledoit_wolf":
        return ledoit_wolf_estimate(
            returns, frequency, currency_convention=currency_convention, **params
        )
    if method == "denoised":
        return denoised(returns, frequency, currency_convention, **params)
    if method == "pca_factor":
        return pca_factor(returns, frequency, currency_convention=currency_convention, **params)
    if method == "group_factor":
        if "groups" not in params or "market" not in params:
            raise ValueError("group_factor needs groups= and market= keyword arguments.")
        return group_factor(
            returns, params.pop("groups"), params.pop("market"), frequency,
            currency_convention, **params,
        )
    if method == "anchored":
        return anchored(returns, frequency, currency_convention=currency_convention, **params)
    raise ValueError(f"Unknown method {method!r}. Available: {', '.join(METHODS)}.")


# --------------------------------------------------------------------------
# Covariance for an allocation
# --------------------------------------------------------------------------

# Methods that produce a correlation on the Pearson scale and so can feed a
# covariance. Spearman and Kendall measure dependence on their own scales
# (tau is about (2/pi) asin r), so plugging them into D C D would mis-state risk.
COVARIANCE_METHODS = (
    "pearson", "ewma", "ledoit_wolf", "denoised", "pca_factor", "group_factor", "anchored",
)

# The trailing window every covariance is estimated on: one trading year, the
# window the walk-forward evaluation (correlations/evaluation.py) tested.
COVARIANCE_WINDOW = 252

# What estimate_covariance uses when the caller does not choose: Ledoit-Wolf
# with the constant-correlation target. In the walk-forward (2026-10-16, both
# configured universes, EUR, 2007-2026) it had the lowest forecast error,
# significantly below the 252-day sample, and tied that sample on the long-only
# minimum-variance portfolio. See docs/PROJECT_CONTEXT.md section 5; every
# correlation report re-runs the test on its own universe.
DEFAULT_COVARIANCE_METHOD = "ledoit_wolf"

PERIODS_PER_YEAR = MappingProxyType({"daily": 252, "weekly": 52})


@dataclass(frozen=True)
class CovarianceEstimate:
    """An annualised covariance matrix and the correlation estimate behind it.

    covariance = D C D, with C from the chosen method and D the sample
    volatilities over the same window. Only C differs between methods, which
    is exactly what the walk-forward evaluation compared.
    """

    covariance: pd.DataFrame  # annualised, in the currency of the returns
    volatility: pd.Series  # annualised
    correlation: CorrelationEstimate
    method: str
    window: int  # observations actually used
    periods_per_year: int
    as_of: pd.Timestamp | None = None
    warnings: tuple[str, ...] = ()

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.covariance.columns)


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = DEFAULT_COVARIANCE_METHOD,
    window: int = COVARIANCE_WINDOW,
    frequency: str = "daily",
    periods_per_year: int | None = None,
    currency_convention: str = "",
    **params,
) -> CovarianceEstimate:
    """Annualised covariance from the last `window` complete rows of `returns`.

    The entry point for the allocation tool. `returns` are log returns, one
    column per asset, as correlations.data.log_returns or
    correlations.analysis.load_returns produce them. Extra keyword arguments go
    to the correlation method, as in estimate_correlation (``target`` for
    ledoit_wolf, ``groups`` and ``market`` for group_factor, ``lam`` for ewma).

    Fewer than `window` rows: the estimate uses what there is and says so.
    Fewer than MIN_OBSERVATIONS: NaN, with the reason. A NaN covariance must
    stop an allocation, never be read as zero risk.
    """
    method = method.strip().lower()
    if method not in COVARIANCE_METHODS:
        raise ValueError(
            f"{method!r} cannot feed a covariance. Use one of {', '.join(COVARIANCE_METHODS)}."
        )
    if window < MIN_OBSERVATIONS:
        raise ValueError(f"window must be at least {MIN_OBSERVATIONS}, got {window}.")
    ppy = int(periods_per_year or PERIODS_PER_YEAR.get(frequency, PERIODS_PER_YEAR["daily"]))

    clean, notes = _prepare(returns)
    tail = clean.iloc[-window:]
    if 0 < len(tail) < window:
        notes.append(f"Only {len(tail)} observations, fewer than the {window}-observation window.")
    if method == "pearson":
        params.setdefault("with_pairs", False)
    if method in HISTORY_METHODS:  # the long run needs every row, the recent part the window
        params.setdefault("window", window)
        correlation = estimate_correlation(clean, method, frequency, currency_convention, **params)
    else:
        correlation = estimate_correlation(tail, method, frequency, currency_convention, **params)

    columns = list(clean.columns)
    volatility = (
        tail.std(ddof=1) * np.sqrt(ppy) if len(tail) >= 2
        else pd.Series(np.nan, index=columns)
    )
    matrix = correlation.matrix.reindex(index=columns, columns=columns).to_numpy()
    vol = volatility.reindex(columns).to_numpy()
    covariance = pd.DataFrame(matrix * np.outer(vol, vol), index=columns, columns=columns)
    warnings = tuple(notes) + tuple(w for w in correlation.warnings if w not in notes)
    if covariance.isna().any().any():
        warnings += ("The covariance has undefined entries; do not size positions from it.",)
    return CovarianceEstimate(
        covariance=covariance,
        volatility=volatility.rename("volatility"),
        correlation=correlation,
        method=method,
        window=int(len(tail)),
        periods_per_year=ppy,
        as_of=tail.index.max() if len(tail) else None,
        warnings=warnings,
    )
