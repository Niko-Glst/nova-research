"""Pearson correlation with its significance, implemented directly.

Used by the correlation module and the backtests. The t distribution's p-value is
computed here rather than pulled from scipy because it is a short continued
fraction, and scipy's DLLs do not always load on this machine. It is checked
against published tables in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Conventional significance threshold, stated once so it is not scattered inline.
SIGNIFICANCE_LEVEL = 0.05


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


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson correlation with its significance."""

    name: str
    correlation: float
    p_value: float
    n_observations: int


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
