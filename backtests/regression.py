"""Ordinary least squares, implemented directly, for validating signal weights.

The pipeline assigns weights to its four analysts -- fundamental 1.0, technical
0.8, insider 0.6, sentiment 0.4 -- chosen by reasoning rather than evidence.
This module provides the machinery to check that choice against realised
returns: regress each signal on the forward return it supposedly predicts, and
look at whether the coefficient is distinguishable from zero.

OLS is implemented here rather than pulled from statsmodels because the whole
computation is a least-squares solve plus its standard errors, and adding a
heavy dependency for that would be a poor trade. numpy's lstsq does the solve.

A warning that belongs in the code rather than only in a README: with the sample
sizes available here (dozens of stocks, one point in time), these regressions are
descriptive, not predictive. An R-squared computed on 30 observations of an
in-sample period tells you what happened, not what will. `summary()` says so
explicitly, and `RegressionResult.is_underpowered` flags it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Below this many observations per predictor, the fit is not worth interpreting.
MIN_OBSERVATIONS_PER_PREDICTOR = 10

# Conventional significance threshold, stated once so it is not scattered inline.
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class Coefficient:
    """One fitted coefficient with its uncertainty."""

    name: str
    value: float
    std_error: float
    t_statistic: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < SIGNIFICANCE_LEVEL

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        margin = 1.96 * self.std_error
        return (self.value - margin, self.value + margin)


@dataclass(frozen=True)
class RegressionResult:
    """A fitted linear model and the diagnostics needed to judge it."""

    coefficients: list[Coefficient]
    r_squared: float
    adjusted_r_squared: float
    n_observations: int
    n_predictors: int
    residual_std_error: float
    f_statistic: float
    f_p_value: float
    target_name: str = "y"

    @property
    def is_underpowered(self) -> bool:
        """Whether there is too little data to interpret the fit."""
        return self.n_observations < MIN_OBSERVATIONS_PER_PREDICTOR * max(
            1, self.n_predictors
        )

    @property
    def significant_predictors(self) -> list[Coefficient]:
        return [c for c in self.coefficients if c.is_significant and c.name != "intercept"]

    def summary(self) -> str:
        """A readable summary, including the caveats that apply to this fit."""
        lines = [
            f"Regression: {self.target_name} ~ "
            + " + ".join(c.name for c in self.coefficients if c.name != "intercept"),
            f"  observations: {self.n_observations}   predictors: {self.n_predictors}",
            f"  R^2: {self.r_squared:.4f}   adjusted R^2: {self.adjusted_r_squared:.4f}",
            f"  residual std error: {self.residual_std_error:.4f}",
            f"  F({self.n_predictors}, {self.n_observations - self.n_predictors - 1}) "
            f"= {self.f_statistic:.3f}, p = {self.f_p_value:.4f}",
            "",
            f"  {'term':<18} {'coef':>10} {'std err':>10} {'t':>8} {'p':>8}",
            f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8}",
        ]
        for coefficient in self.coefficients:
            marker = " *" if coefficient.is_significant else ""
            lines.append(
                f"  {coefficient.name:<18} {coefficient.value:>10.4f} "
                f"{coefficient.std_error:>10.4f} {coefficient.t_statistic:>8.3f} "
                f"{coefficient.p_value:>8.4f}{marker}"
            )

        lines.append("")
        if self.is_underpowered:
            lines.append(
                f"  WARNING: {self.n_observations} observations for "
                f"{self.n_predictors} predictor(s) is below the "
                f"{MIN_OBSERVATIONS_PER_PREDICTOR}:1 rule of thumb. Treat every "
                f"coefficient here as descriptive, not predictive."
            )
        lines.append(
            "  Note: a fit on historical data describes what happened. It is not "
            "evidence that the relationship will hold out of sample."
        )
        return "\n".join(lines)


def _student_t_sf(t: float, degrees_of_freedom: int) -> float:
    """Two-sided survival function for Student's t, without scipy.

    Uses the regularized incomplete beta function, which is what the t
    distribution's CDF reduces to. Accurate enough for reporting p-values.
    """
    if degrees_of_freedom <= 0:
        return float("nan")
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    return _regularized_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b), via the continued fraction in Numerical Recipes."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    # The continued fraction converges quickly for x < (a+1)/(a+b+2); otherwise
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_beta) * _beta_continued_fraction(x, a, b) / a
    return 1.0 - math.exp(log_beta) * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_continued_fraction(x: float, a: float, b: float, iterations: int = 200) -> float:
    """Lentz's algorithm for the beta continued fraction."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, iterations + 1):
        m2 = 2 * m
        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < 3e-16:
            break

    return result


def _f_distribution_sf(f: float, d1: int, d2: int) -> float:
    """Survival function for the F distribution, via the incomplete beta."""
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return _regularized_incomplete_beta(x, d2 / 2.0, d1 / 2.0)


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    predictor_names: list[str] | None = None,
    target_name: str = "y",
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit y = Xb + e by ordinary least squares.

    X is (n_observations, n_predictors); y is (n_observations,). Rows containing
    NaN in either are dropped, since a missing signal is not a zero signal.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}; they must match."
        )

    # Drop incomplete rows rather than imputing: an imputed signal would be
    # indistinguishable from a measured neutral one.
    complete = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[complete], y[complete]

    n_observations = X.shape[0]
    n_predictors = X.shape[1]

    if n_observations <= n_predictors + (1 if add_intercept else 0):
        raise ValueError(
            f"Not enough observations ({n_observations}) to fit {n_predictors} "
            f"predictor(s). Need at least {n_predictors + 2}."
        )

    names = list(predictor_names) if predictor_names else [
        f"x{i + 1}" for i in range(n_predictors)
    ]
    if len(names) != n_predictors:
        raise ValueError(
            f"predictor_names has {len(names)} entries for {n_predictors} predictors."
        )

    design = np.column_stack([np.ones(n_observations), X]) if add_intercept else X
    all_names = (["intercept"] + names) if add_intercept else names

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted

    degrees_of_freedom = n_observations - design.shape[1]
    residual_sum_of_squares = float(residuals @ residuals)
    total_sum_of_squares = float(((y - y.mean()) ** 2).sum())

    r_squared = (
        1.0 - residual_sum_of_squares / total_sum_of_squares
        if total_sum_of_squares > 0
        else 0.0
    )
    adjusted = (
        1.0 - (1.0 - r_squared) * (n_observations - 1) / degrees_of_freedom
        if degrees_of_freedom > 0
        else float("nan")
    )

    sigma_squared = (
        residual_sum_of_squares / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    )
    residual_std_error = math.sqrt(sigma_squared) if sigma_squared == sigma_squared else float("nan")

    # Standard errors from the diagonal of sigma^2 (X'X)^-1, via pseudo-inverse
    # so a collinear design degrades rather than raising.
    covariance = sigma_squared * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    coefficients: list[Coefficient] = []
    for index, name in enumerate(all_names):
        value = float(beta[index])
        std_error = float(standard_errors[index])
        if std_error > 0:
            t_statistic = value / std_error
            p_value = _student_t_sf(t_statistic, degrees_of_freedom)
        else:
            t_statistic, p_value = float("nan"), float("nan")
        coefficients.append(
            Coefficient(
                name=name,
                value=value,
                std_error=std_error,
                t_statistic=t_statistic,
                p_value=p_value,
            )
        )

    # Overall F test: does the model explain more than the mean alone?
    if n_predictors > 0 and degrees_of_freedom > 0 and total_sum_of_squares > 0:
        explained = total_sum_of_squares - residual_sum_of_squares
        f_statistic = (explained / n_predictors) / sigma_squared
        f_p_value = _f_distribution_sf(f_statistic, n_predictors, degrees_of_freedom)
    else:
        f_statistic, f_p_value = float("nan"), float("nan")

    return RegressionResult(
        coefficients=coefficients,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        n_observations=n_observations,
        n_predictors=n_predictors,
        residual_std_error=residual_std_error,
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        target_name=target_name,
    )


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson correlation with its significance."""

    name: str
    correlation: float
    p_value: float
    n_observations: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < SIGNIFICANCE_LEVEL


def correlate(x: np.ndarray, y: np.ndarray, name: str = "x") -> CorrelationResult:
    """Pearson correlation between two series, with a t-based p-value."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    complete = ~(np.isnan(x) | np.isnan(y))
    x, y = x[complete], y[complete]
    n = len(x)

    if n < 3 or x.std() == 0 or y.std() == 0:
        return CorrelationResult(name=name, correlation=float("nan"), p_value=float("nan"), n_observations=n)

    r = float(np.corrcoef(x, y)[0, 1])
    # Guard the degenerate |r| == 1 case, where the t statistic diverges.
    if abs(r) >= 1.0:
        return CorrelationResult(name=name, correlation=r, p_value=0.0, n_observations=n)

    t = r * math.sqrt((n - 2) / (1 - r * r))
    return CorrelationResult(
        name=name, correlation=r, p_value=_student_t_sf(t, n - 2), n_observations=n
    )
