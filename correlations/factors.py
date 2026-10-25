"""Factor-model correlation matrices.

A factor model writes each return as exposure to a few common factors plus an
idiosyncratic part uncorrelated with everything else:

    r_i = sum_k B_ik f_k + e_i     =>     C = B F B' + D

Every pairwise correlation is then explained by shared factors, so N(N-1)/2
noisy pairwise estimates collapse into N x K loadings. That is the idea behind
commercial risk models (MSCI Barra, Axioma), which combine a market factor,
industry/sector factors and style factors. Two versions here:

- **Statistical (PCA):** the factors are the leading principal components of
  the correlation matrix, K chosen as the fewest components explaining at
  least PCA_VARIANCE_TARGET of the variance. No labels needed, but the factors
  have no names.
- **Market + group:** one market factor (the universe's proxy, e.g. SPY) and
  one factor per group label from the config (asset class or sector), each the
  equal-weighted market-residual return of the group's members. This is the
  Barra structure in miniature, with interpretable factors.

Both return a positive semi-definite matrix with a unit diagonal by
construction: B F B' is PSD and D is a non-negative diagonal.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Share of total variance the PCA factors must explain. 80% is a common rule
# of thumb for a statistical risk model: enough components to capture the
# shared structure, few enough that the rest is plausibly idiosyncratic.
PCA_VARIANCE_TARGET = 0.80

# A group needs at least this many members to get its own factor. A group of
# one would make the factor equal to that asset's own residual, explaining it
# perfectly and adding no shared structure.
MIN_GROUP_SIZE = 2


@dataclass(frozen=True)
class FactorModelResult:
    correlation: np.ndarray
    covariance: np.ndarray  # in the units of the input returns
    n_factors: int
    explained: float  # share of total variance explained by the factors
    factor_names: tuple[str, ...]
    loadings: np.ndarray  # N x K
    warnings: tuple[str, ...] = ()


def _to_correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(covariance))
    corr = covariance / np.outer(scale, scale)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def pca_factor_model(
    returns: np.ndarray, variance_target: float = PCA_VARIANCE_TARGET
) -> FactorModelResult:
    """Statistical factor model from the leading principal components.

    Works on the correlation matrix (standardised returns), so a high-volatility
    asset does not dominate the components merely by being volatile.
    """
    x = np.asarray(returns, dtype=float)
    std = x.std(axis=0, ddof=1)
    corr = np.corrcoef(x, rowvar=False)
    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]

    cumulative = np.cumsum(values) / values.sum()
    k = int(np.searchsorted(cumulative, variance_target) + 1)
    k = min(k, len(values))
    loadings = vectors[:, :k] * np.sqrt(values[:k])

    common = loadings @ loadings.T
    specific = np.clip(1.0 - np.diag(common), 0.0, None)
    model_corr = common + np.diag(specific)
    model_corr = _to_correlation(model_corr)
    covariance = model_corr * np.outer(std, std)

    warnings: list[str] = []
    if k == len(values):
        warnings.append(
            f"Reaching {variance_target:.0%} explained variance needed every component, so "
            f"the factor model reproduces the sample matrix."
        )
    return FactorModelResult(
        correlation=model_corr,
        covariance=covariance,
        n_factors=k,
        explained=float(cumulative[k - 1]),
        factor_names=tuple(f"PC{i + 1}" for i in range(k)),
        loadings=loadings,
        warnings=tuple(warnings),
    )


def market_group_factor_model(
    returns: pd.DataFrame, groups: Mapping[str, str], market: pd.Series
) -> FactorModelResult:
    """Market + group factor model, estimated in covariance form.

    1. Regress every asset on the market return: beta_i and residual e_i.
    2. Within each group, set every residual correlation to the group's
       average residual correlation; across groups, set it to zero. This is
       exactly the residual structure a market + one-factor-per-group model
       implies when members load equally on their group.
    3. Covariance = beta beta' var(market) + S R S, with S the diagonal of
       residual volatilities and R the block matrix from step 2. Both terms are
       positive semi-definite (a constant-correlation block is PSD for any
       average at or above -1/(n-1), where it is clipped), so the sum is too.

    Why not a time-series group factor (the average residual of the group's
    members)? With the 2-6 members these groups have, that proxy is too noisy:
    including the asset itself overstated a true within-group correlation of
    0.82 as 0.88, and leaving it out understated it as 0.72 (errors in
    variables). Estimating the group effect as an average correlation avoids
    the proxy altogether.

    Groups smaller than MIN_GROUP_SIZE are folded into "other"; an asset in a
    group that still has fewer members has no group effect.
    """
    frame = returns.astype(float)
    market = market.reindex(frame.index).astype(float)
    if market.isna().any():
        keep = market.notna()
        frame, market = frame[keep], market[keep]
    assets = list(frame.columns)
    m = market.to_numpy()
    x = frame.to_numpy()
    t = len(m)

    warnings: list[str] = []
    labels = {a: groups.get(a, "other") for a in assets}
    sizes = pd.Series(labels).value_counts()
    small = [g for g, size in sizes.items() if size < MIN_GROUP_SIZE]
    if small:
        warnings.append(
            f"Group(s) with a single member folded into 'other': {', '.join(sorted(small))}."
        )
        labels = {a: ("other" if g in small else g) for a, g in labels.items()}
    sizes = pd.Series(labels).value_counts()

    # Step 1: market betas and residuals.
    design = np.column_stack([np.ones(t), m])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    beta = coef[1]
    residuals = x - design @ coef
    resid_std = residuals.std(axis=0, ddof=2)
    market_var = float(m.var(ddof=1))

    group_names = sorted(g for g, size in sizes.items() if size >= MIN_GROUP_SIZE)
    names = ("market",) + tuple(group_names)
    k = len(names)

    # Step 2: block residual correlation, constant within each group.
    n = len(assets)
    resid_corr = np.eye(n)
    loadings = np.zeros((n, k))
    loadings[:, 0] = beta * np.sqrt(market_var)
    group_variance = np.zeros(n)
    for g_index, group in enumerate(group_names):
        idx = [i for i, a in enumerate(assets) if labels[a] == group]
        usable = [i for i in idx if resid_std[i] > 0]  # the proxy itself has no residual
        if len(usable) < 2:
            continue
        block = np.corrcoef(residuals[:, usable], rowvar=False)
        size = len(usable)
        average = float((block.sum() - size) / (size * (size - 1)))
        average = max(average, -1.0 / (size - 1))
        for a in usable:
            for b in usable:
                if a != b:
                    resid_corr[a, b] = average
        if average > 0:
            loadings[usable, 1 + g_index] = np.sqrt(average) * resid_std[usable]
            group_variance[usable] = average * resid_std[usable] ** 2

    # Step 3: market part plus residual part.
    covariance = np.outer(beta, beta) * market_var + np.outer(resid_std, resid_std) * resid_corr
    total_var = x.var(axis=0, ddof=1).sum()
    explained = float(((beta**2) * market_var + group_variance).sum() / total_var)

    return FactorModelResult(
        correlation=_to_correlation(covariance),
        covariance=covariance,
        n_factors=k,
        explained=explained,
        factor_names=names,
        loadings=loadings,
        warnings=tuple(warnings),
    )
