"""Random matrix theory: separate signal from noise in a correlation matrix.

If N assets were truly uncorrelated, the eigenvalues of their sample
correlation matrix from T observations would not all equal 1. They would
spread over the Marchenko-Pastur band [lambda_minus, lambda_plus], with

    lambda_+- = sigma^2 * (1 +- sqrt(N / T))^2

purely from sampling error. Eigenvalues inside that band are
indistinguishable from noise; only those above it carry structure.

Denoising (constant residual eigenvalue method, Lopez de Prado, *Machine
Learning for Asset Managers*, 2020, ch. 2):

1. Test the largest eigenvalue against pure noise (sigma^2 = 1). If it does
   not clear the noise edge, nothing is signal and the matrix is left as is.
2. Otherwise estimate the noise variance as sigma^2 = 1 - lambda_max / N:
   the share of total variance not taken by the market mode (Laloux, Cizeau,
   Bouchaud & Potters 1999, "Noise dressing of financial correlation
   matrices").
3. Count eigenvalues above the finite-sample noise edge as signal. The
   asymptotic lambda_plus holds for infinitely many assets; for finite N the
   largest noise eigenvalue fluctuates around it (Tracy-Widom), so the edge
   used is the Tracy-Widom 95th percentile (Johnstone 2001).
4. Replace the noise eigenvalues by their average, which keeps the trace (the
   total variance) and removes the spurious spread; rebuild the matrix and
   rescale it to a unit diagonal.

Why not fit sigma^2 to the eigenvalue density, as Lopez de Prado does? That
fit was implemented and tested first. On synthetic data it was exact, but on
the real 30-asset universe it hit the upper bound sigma^2 = 1 in every window
tried (full sample, the last year, 2008, 2017): one market eigenvalue of 10-18
followed by a smooth decay matches no single-sigma^2 band, so the density fit
has nothing to lock onto. Iterating sigma^2 as the mean of the remaining
eigenvalues collapses the other way (24-28 of 30 "signal"). The Laloux
estimator gives 8 signal eigenvalues on 20 years of daily data (T/N = 163,
precise) and 4-5 in one-year windows (T/N = 8, noisy), which is the pattern
estimation error should produce. In simulation it recovers 3 and 5 planted
factors exactly in 20 of 20 draws, and with the step-1 test it reports no
false signal in 40 pure-noise draws.

Known limitations: Marchenko-Pastur assumes every asset has the same share of
idiosyncratic variance, and real assets do not. With N = 30 the separation of
signal from noise is approximate however it is done. How much denoising
changes the matrix depends on T / N: little on decades of daily data, a lot on
a one-year window, which is where the walk-forward evaluation (phase 3) shows
whether it helps.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this many assets the eigenvalue spectrum is too short for the
# signal/noise separation to say much, and the result says so.
RMT_MIN_ASSETS = 20

# 95th percentile of the Tracy-Widom distribution for real-valued data
# (beta = 1), from the published tables (Tracy & Widom 1996). An eigenvalue
# above the corresponding edge has under a 5% chance of being the largest
# eigenvalue of pure noise.
TRACY_WIDOM_95 = 0.9793

# Bandwidth factor for the eigenvalue density drawn in the report (Silverman's
# robust rule). Used for display only; it plays no part in the denoising.
SILVERMAN_FACTOR = 0.9


@dataclass(frozen=True)
class DenoiseResult:
    correlation: np.ndarray
    eigenvalues: np.ndarray  # of the input matrix, descending
    denoised_eigenvalues: np.ndarray  # descending, before rescaling
    sigma2: float
    lambda_minus: float
    lambda_plus: float
    n_signal: int
    q: float  # T / N
    warnings: tuple[str, ...] = ()
    signal_edge: float = float("nan")  # Tracy-Widom 95% edge used to count signal


def marchenko_pastur_bounds(sigma2: float, q: float) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur band for variance sigma2 and T/N = q."""
    root = math.sqrt(1.0 / q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def marchenko_pastur_pdf(x: np.ndarray, sigma2: float, q: float) -> np.ndarray:
    """Marchenko-Pastur density at the points `x` (zero outside the band)."""
    lo, hi = marchenko_pastur_bounds(sigma2, q)
    x = np.asarray(x, dtype=float)
    inside = (x > lo) & (x < hi)
    pdf = np.zeros_like(x)
    xi = x[inside]
    pdf[inside] = q / (2.0 * math.pi * sigma2 * xi) * np.sqrt((hi - xi) * (xi - lo))
    return pdf


def tracy_widom_edge(sigma2: float, n_observations: int, n_assets: int) -> float:
    """95% finite-sample upper edge of the noise eigenvalues (Johnstone 2001).

    For T x N white noise, the largest eigenvalue of X'X centres on
    mu = (sqrt(T - 1) + sqrt(N))^2 with scale
    s = (sqrt(T - 1) + sqrt(N)) * (1/sqrt(T - 1) + 1/sqrt(N))^(1/3);
    dividing by T puts it on the correlation-matrix scale.
    """
    a, b = math.sqrt(n_observations - 1), math.sqrt(n_assets)
    mu = (a + b) ** 2
    scale = (a + b) * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    return sigma2 * (mu + TRACY_WIDOM_95 * scale) / n_observations


def noise_variance(eigenvalues: np.ndarray, n_observations: int) -> float:
    """sigma^2 of the noise: 1 if the top eigenvalue is noise, else 1 - lambda_max / N."""
    values = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    n = len(values)
    if values[0] <= tracy_widom_edge(1.0, n_observations, n):
        return 1.0
    return float(max(1.0 - values[0] / n, 1e-6))


def gaussian_kde(samples: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel density estimate of `samples`, evaluated at `points`."""
    samples = np.asarray(samples, dtype=float)[:, None]
    z = (np.asarray(points, dtype=float)[None, :] - samples) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=0) / (len(samples) * bandwidth * math.sqrt(2 * math.pi))


def silverman_bandwidth(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    spread = samples.std(ddof=1)
    q75, q25 = np.percentile(samples, [75, 25])
    robust = min(spread, (q75 - q25) / 1.34) or spread
    return float(SILVERMAN_FACTOR * robust * len(samples) ** (-0.2)) or 1e-3


def denoise(correlation: np.ndarray, n_observations: int) -> DenoiseResult:
    """Constant-residual-eigenvalue denoising of a correlation matrix."""
    corr = np.asarray(correlation, dtype=float)
    n = corr.shape[0]
    if np.isnan(corr).any():
        raise ValueError("Cannot denoise a matrix containing NaN.")
    q = n_observations / n
    warnings: list[str] = []
    if q <= 1.0:
        raise ValueError(f"T / N = {q:.2f}: need more observations than assets to denoise.")

    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    sigma2 = noise_variance(values, n_observations)
    lo, hi = marchenko_pastur_bounds(sigma2, q)
    edge = tracy_widom_edge(sigma2, n_observations, n)
    n_signal = int((values > edge).sum())

    if n < RMT_MIN_ASSETS:
        warnings.append(
            f"Only {n} assets: too short a spectrum to separate signal from noise reliably. "
            f"Random matrix theory needs a broad universe to say much."
        )
    if n_signal == n:
        warnings.append("Every eigenvalue sits above the noise edge; nothing was denoised.")

    denoised = values.copy()
    if n_signal < n:
        denoised[n_signal:] = values[n_signal:].mean()
    rebuilt = (vectors * denoised) @ vectors.T
    scale = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(scale, scale)
    rebuilt = (rebuilt + rebuilt.T) / 2.0
    np.fill_diagonal(rebuilt, 1.0)

    return DenoiseResult(
        correlation=rebuilt,
        eigenvalues=values,
        denoised_eigenvalues=denoised,
        sigma2=sigma2,
        lambda_minus=lo,
        lambda_plus=hi,
        n_signal=n_signal,
        q=q,
        warnings=tuple(warnings),
        signal_edge=edge,
    )
