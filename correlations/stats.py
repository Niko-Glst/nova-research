"""Statistics the correlation estimators need.

Pearson significance already exists in backtests/regression.py; this module
adds what that one does not cover:

- **Kendall's tau-b** with the tie correction and its asymptotic p-value.
  Returns have ties (exact repeats, zero days), and the uncorrected tau-a
  understates dependence when they occur. scipy is an optional dependency
  (PROJECT_CONTEXT.md section 8): when it loads, `kendall_matrix` uses
  scipy.stats.kendalltau, an O(n log n) merge-sort count that took 0.3 s for
  30 assets x 5,000 days against 16-200 s for the O(n^2) version here. When it
  does not load (not installed, or a compiled file blocked by Windows Smart
  App Control), the built-in version runs instead and `scipy_status()` says
  why, so the report can tell the user how to fix it. The two agree to 1e-12
  on tau and the p-value (tests/test_correlation_stats.py).
- **Benjamini-Hochberg.** An N-asset matrix tests N(N-1)/2 pairs. At p < 0.05,
  about one pair in twenty is "significant" by chance alone, so a matrix of 55
  pairs produces two or three false findings even when nothing is related. BH
  controls the expected share of false discoveries among the pairs called
  significant, which is the right error rate for scanning a matrix.
- **A positive semi-definiteness check.** A correlation matrix that is not PSD
  implies a portfolio with negative variance, and any optimiser fed one will
  exploit that.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from backtests.regression import SIGNIFICANCE_LEVEL

# What to run when scipy is missing or blocked. 1.18.1 is the version that
# loaded under Smart App Control on the owner's machine (2026-10-16); a newer
# release ships new compiled files, which Smart App Control may block again.
SCIPY_FIX = r".venv\Scripts\python.exe -m pip install scipy==1.18.1"

# Rows compared per block in the O(n^2) Kendall computation. 512 rows against
# 5,000 observations is about 2.5M int8 signs per asset: small enough for any
# machine, large enough that the Python loop overhead is negligible.
KENDALL_CHUNK = 512

# Rows per block in the matrix form of kendall_matrix. Each block holds
# N x rows x T float32 signs (30 x 128 x 5,000 is about 77 MB), and a block's
# per-pair partial sum is at most rows x T = 640,000, far below the 2^24 up to
# which float32 counts exactly.
KENDALL_BLAS_ROWS = 128

# Eigenvalues above -1e-10 count as zero. Floating point leaves residue of that
# order on a matrix that is PSD by construction, and calling that a violation
# would make every exact result fail its own check.
PSD_TOLERANCE = 1e-10

# A diagonal entry this far from 1 means the matrix is not a correlation matrix.
DIAGONAL_TOLERANCE = 1e-8


# --------------------------------------------------------------------------
# Kendall's tau-b
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KendallResult:
    tau: float
    p_value: float
    n_observations: int


def _tie_sums(values: np.ndarray) -> tuple[float, float, float]:
    """Sums over tie groups of t(t-1), t(t-1)(t-2) and t(t-1)(2t+5)."""
    _, counts = np.unique(values, return_counts=True)
    t = counts[counts > 1].astype(float)
    return (
        float((t * (t - 1)).sum()),
        float((t * (t - 1) * (t - 2)).sum()),
        float((t * (t - 1) * (2 * t + 5)).sum()),
    )


def _upper_signs(column: np.ndarray, start: int, stop: int) -> np.ndarray:
    """sign(x_j - x_i) for rows i in [start, stop) against all j, zeroed for j <= i."""
    n = column.size
    signs = np.sign(column[None, :] - column[start:stop, None]).astype(np.int8)
    rows = np.arange(start, stop)[:, None]
    signs[np.arange(n)[None, :] <= rows] = 0
    return signs


def _kendall_from_s(s: float, x: np.ndarray, y: np.ndarray) -> KendallResult:
    """tau-b and its p-value from the concordance sum S = C - D.

    The variance of S under independence, with ties in both variables, is the
    standard expression (Kendall 1970, Rank Correlation Methods, ch. 4), the
    same one scipy uses for its asymptotic method.
    """
    n = x.size
    n0 = n * (n - 1) / 2.0
    tx1, tx2, tx5 = _tie_sums(x)
    ty1, ty2, ty5 = _tie_sums(y)
    n1, n2 = tx1 / 2.0, ty1 / 2.0

    denominator = math.sqrt((n0 - n1) * (n0 - n2))
    if denominator == 0.0:
        return KendallResult(float("nan"), float("nan"), n)
    tau = s / denominator

    variance = (
        (n * (n - 1) * (2 * n + 5) - tx5 - ty5) / 18.0
        + (tx2 * ty2) / (9.0 * n * (n - 1) * (n - 2))
        + (tx1 * ty1) / (2.0 * n * (n - 1))
    )
    if variance <= 0.0:
        return KendallResult(float(tau), float("nan"), n)
    z = s / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return KendallResult(float(tau), float(p_value), n)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> KendallResult:
    """Kendall's tau-b between two series, with an asymptotic two-sided p-value.

    Rows with a NaN in either series are dropped first.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    complete = ~(np.isnan(x) | np.isnan(y))
    x, y = x[complete], y[complete]
    n = x.size
    if n < 3:
        return KendallResult(float("nan"), float("nan"), n)

    s = 0
    for start in range(0, n, KENDALL_CHUNK):
        stop = min(n, start + KENDALL_CHUNK)
        s += int(
            (_upper_signs(x, start, stop) * _upper_signs(y, start, stop)).sum(dtype=np.int64)
        )
    return _kendall_from_s(float(s), x, y)


@lru_cache(maxsize=1)
def _scipy_kendalltau() -> tuple[Callable | None, str]:
    """scipy.stats.kendalltau, or None and the reason it could not be loaded."""
    try:
        from scipy.stats import kendalltau
    except Exception as exc:  # ImportError, or a DLL blocked by Windows application control
        return None, f"{type(exc).__name__}: {exc}"
    return kendalltau, ""


def scipy_status() -> tuple[bool, str]:
    """(available, reason it is not) for the optional scipy dependency."""
    function, reason = _scipy_kendalltau()
    return function is not None, reason


def kendall_backend() -> str:
    """Which Kendall implementation `kendall_matrix` will use: "scipy" or "built-in"."""
    return "scipy" if scipy_status()[0] else "built-in"


def kendall_matrix(
    values: np.ndarray, use_scipy: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """tau-b and p-value matrices for the columns of a complete (no NaN) array.

    With scipy available (and `use_scipy`), each pair goes through
    scipy.stats.kendalltau with the asymptotic p-value, the same tie-corrected
    variance as `_kendall_from_s`. Otherwise the built-in version runs: every
    pair's concordance sum S_ab = sum over i<j of sign(dx_a) sign(dx_b) is an
    inner product of two sign vectors, so all of them at once are one matrix
    product M M' per block of rows, done by BLAS. The products are exact: each
    block's partial sums are integers below 2^24, which float32 represents
    without rounding, and blocks are accumulated in int64. (Multiplying the
    pairs one by one took 62 s for 30 assets and 4,900 days.)
    """
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    if np.isnan(values).any():
        raise ValueError("kendall_matrix needs complete data; drop NaN rows first.")

    kendalltau, _ = _scipy_kendalltau()
    if use_scipy and kendalltau is not None:
        tau = np.eye(k)
        p_values = np.zeros((k, k))
        for a in range(k):
            for b in range(a + 1, k):
                result = kendalltau(values[:, a], values[:, b], method="asymptotic")
                tau[a, b] = tau[b, a] = float(result.statistic)
                p_values[a, b] = p_values[b, a] = float(result.pvalue)
        return tau, p_values

    concordance = np.zeros((k, k), dtype=np.int64)
    columns = values.T.astype(np.float64)  # k x n
    for start in range(0, n, KENDALL_BLAS_ROWS):
        stop = min(n, start + KENDALL_BLAS_ROWS)
        # Only pairs j > i count; one mask for the block, shared by all columns.
        upper = (np.arange(n)[None, :] > np.arange(start, stop)[:, None]).astype(np.float32)
        diffs = columns[:, None, :] - columns[:, start:stop, None]  # k x rows x n
        block = (np.sign(diffs).astype(np.float32) * upper).reshape(k, -1)
        concordance += np.rint(block @ block.T).astype(np.int64)

    tau = np.eye(k)
    p_values = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            result = _kendall_from_s(float(concordance[a, b]), values[:, a], values[:, b])
            tau[a, b] = tau[b, a] = result.tau
            p_values[a, b] = p_values[b, a] = result.p_value
    return tau, p_values


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float = SIGNIFICANCE_LEVEL
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjusted p-values and rejections.

    Adjusted p for the i-th smallest of m p-values is min over j >= i of
    p_(j) * m / j, capped at 1. A pair is significant after correction when its
    adjusted p is at most `alpha`. NaN p-values (pairs that could not be
    tested) stay NaN and do not count toward m: an untested pair is not a test.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted, np.zeros(p.shape, dtype=bool)

    observed = p[valid]
    order = np.argsort(observed)
    ranked = observed[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(ranked, 1.0)

    adjusted[valid] = result
    reject = np.zeros(p.shape, dtype=bool)
    reject[valid] = result <= alpha
    return adjusted, reject


# --------------------------------------------------------------------------
# Matrix checks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PsdCheck:
    is_psd: bool
    min_eigenvalue: float
    unit_diagonal: bool
    warnings: tuple[str, ...] = ()


def check_psd(matrix: np.ndarray) -> PsdCheck:
    """Whether a matrix is a valid correlation matrix: symmetric PSD, unit diagonal."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0 or np.isnan(matrix).any():
        return PsdCheck(
            False, float("nan"), False, ("Matrix contains NaN, so it is not a usable estimate.",)
        )

    warnings: list[str] = []
    symmetric = np.allclose(matrix, matrix.T, atol=1e-12)
    if not symmetric:
        warnings.append("Matrix is not symmetric.")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    minimum = float(eigenvalues.min())
    is_psd = symmetric and minimum >= -PSD_TOLERANCE
    if minimum < -PSD_TOLERANCE:
        warnings.append(
            f"Smallest eigenvalue is {minimum:.2e}: the matrix is not positive semi-definite, "
            f"so it implies a portfolio with negative variance."
        )
    unit = bool(np.all(np.abs(np.diag(matrix) - 1.0) <= DIAGONAL_TOLERANCE))
    if not unit:
        warnings.append("Diagonal is not all ones.")
    return PsdCheck(is_psd, minimum, unit, tuple(warnings))
