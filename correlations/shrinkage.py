"""Ledoit-Wolf shrinkage of the covariance matrix, written in numpy.

A sample covariance matrix estimated from T observations of N assets has
N(N+1)/2 free parameters. When T is not much larger than N, its extreme
eigenvalues are systematically too extreme: the largest too large, the
smallest too small. An optimiser fed such a matrix finds "riskless"
combinations that are artefacts of estimation error and bets on them.

Shrinkage pulls the sample matrix S toward a structured target F:

    shrunk = delta * F + (1 - delta) * S

with the intensity delta chosen in closed form to minimise expected squared
(Frobenius) error. Two targets are implemented:

- **Constant correlation** (Ledoit & Wolf 2004, "Honey, I shrunk the sample
  covariance matrix", J. Portfolio Management): keeps each asset's own
  variance and replaces every correlation by the average correlation. This is
  the default. Its structure, one common correlation, is a reasonable prior
  for assets that share a market factor, and it leaves the variances alone, so
  shrinkage changes only the dependence the module is about.
- **Scaled identity** (Ledoit & Wolf 2004, "A well-conditioned estimator for
  large-dimensional covariance matrices", J. Multivariate Analysis): pulls
  toward mu * I, every variance the average and every covariance zero. It
  shrinks correlations toward zero, the more conservative prior for a
  cross-asset universe where some pairs are genuinely negative (equities vs
  treasuries), which the constant-correlation target would pull toward the
  positive average.

Which target forecasts better is an empirical question; the walk-forward
evaluation in phase 3 answers it on this data.

Both follow the published estimators exactly. The identity version is checked
against scikit-learn's `ledoit_wolf` and the constant-correlation version
against PyPortfolioOpt (a port of Ledoit and Wolf's own covCor.m), with the
reference values hardcoded in the tests. Neither library is a dependency.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGETS = ("constant_correlation", "identity")


@dataclass(frozen=True)
class ShrinkageResult:
    covariance: np.ndarray  # shrunk, in the units of the input returns
    intensity: float  # delta in [0, 1]: weight on the target
    target: str
    average_correlation: float = float("nan")  # constant-correlation target only


def _centered(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("returns must be a 2-D array with at least two rows.")
    return values - values.mean(axis=0)


def shrink_identity(returns: np.ndarray) -> ShrinkageResult:
    """Ledoit-Wolf (2004) shrinkage toward mu * I.

    S is the maximum-likelihood covariance (divide by T), as in the paper.
    The intensity is beta / delta, where delta is the squared distance between
    S and the target and beta estimates how much of that distance is sampling
    error, capped at delta.
    """
    x = _centered(returns)
    t, n = x.shape
    sample = x.T @ x / t
    mu = float(np.trace(sample)) / n

    delta = float(((sample - mu * np.eye(n)) ** 2).sum()) / n
    x2 = x**2
    beta_raw = float(((x2.T @ x2) / t - sample**2).sum()) / (n * t)
    beta = min(beta_raw, delta)
    intensity = 0.0 if delta == 0.0 else beta / delta

    shrunk = (1.0 - intensity) * sample + intensity * mu * np.eye(n)
    return ShrinkageResult(shrunk, float(intensity), "identity")


def shrink_constant_correlation(returns: np.ndarray) -> ShrinkageResult:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation matrix.

    Notation follows the paper's appendix: pi is the summed asymptotic variance
    of the sample covariances, rho the part of it shared with the target, gamma
    the misspecification of the target. The optimal intensity is
    kappa / T with kappa = (pi - rho) / gamma, clipped to [0, 1].
    """
    x = _centered(returns)
    t, n = x.shape
    sample = x.T @ x / t
    var = np.diag(sample).copy()
    std = np.sqrt(var)

    if np.any(std == 0.0):
        raise ValueError("A column has zero variance; constant-correlation shrinkage is undefined.")

    outer_std = np.outer(std, std)
    corr = sample / outer_std
    rbar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = rbar * outer_std
    np.fill_diagonal(target, var)

    # pi: sum over i, j of Var(sqrt(T) * s_ij).
    y = x**2
    pi_matrix = (y.T @ y) / t - sample**2
    pi_hat = float(pi_matrix.sum())

    # rho: diagonal part plus the off-diagonal covariance with the target.
    rho_diag = float(np.trace(pi_matrix))
    theta = ((x**3).T @ x) / t - var[:, None] * sample  # theta_ii,ij, row i
    off = std[None, :] / std[:, None] * theta  # sqrt(s_jj / s_ii) * theta_ii,ij
    rho_off = rbar * float((off + off.T).sum() - 2.0 * np.trace(off)) / 2.0
    rho_hat = rho_diag + rho_off

    gamma_hat = float(((sample - target) ** 2).sum())
    if gamma_hat == 0.0:
        intensity = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        intensity = float(max(0.0, min(1.0, kappa / t)))

    shrunk = intensity * target + (1.0 - intensity) * sample
    return ShrinkageResult(shrunk, intensity, "constant_correlation", float(rbar))


def ledoit_wolf(returns: np.ndarray, target: str = "constant_correlation") -> ShrinkageResult:
    if target == "constant_correlation":
        return shrink_constant_correlation(returns)
    if target == "identity":
        return shrink_identity(returns)
    raise ValueError(f"Unknown shrinkage target {target!r}; use one of {TARGETS}.")
