"""Weighted-signal strategy: evaluation and weight fitting, in linear algebra.

The strategy, for every stock n on every day t::

    score[t, n]    = X[t-1, n, :] @ w                (yesterday's signals, weighted)
    position[t, n] = 1 if score[t, n] > theta else 0 (long or cash, never short)

and the portfolio holds 1/N of capital in each stock's strategy, rebalanced
daily, paying FEE_BPS on every change of position. Buy & hold of the same
stocks, equally weighted, is the benchmark. The current pipeline strategy (the
50/200-day crossover) is the special case w = trend only, theta = 0.

Panel layout: X has shape (T, N, K) -- days, stocks, signals -- already lagged
one day, so X[t] holds signals known at the previous close. R has shape
(T, N): the close-to-close simple return of day t. Scores for many weight
vectors at once are one tensor product, X @ W with W of shape (K, M).

Four ways to choose w, all fitted on a training window only:

- **ridge:** regress each stock's next-day return (scaled by its own training
  volatility, so a volatile stock does not dominate) on the signals:
  w = (X'X + lambda I)^-1 X'y, with the intercept left unpenalised. Long when
  the predicted return is positive. The regression asks "which combination of
  signals best predicts tomorrow's return".
- **max_sharpe:** treat each signal as a trading strategy of its own (position
  proportional to the signal) and find the mix with the highest Sharpe ratio,
  the tangency portfolio w ~ Sigma^-1 mu. Sigma is Ledoit-Wolf shrunk
  (correlations/shrinkage.py) because signal returns are noisy and correlated.
- **search:** random weight vectors and thresholds, keeping the one with the
  best training Sharpe after costs. The most flexible, and the most prone to
  fitting noise; the deflated Sharpe ratio in walk_forward.py accounts for the
  number of tries.
- **equal:** every signal weighted equally, nothing fitted. If a fitted method
  cannot beat this out of sample, the fitting adds nothing.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from correlations.shrinkage import shrink_identity

TRADING_DAYS = 252

# Cost per change of position, per side, in basis points. Matches the 10 bps
# the existing backtest charges.
FEE_BPS = 10.0

# Ridge penalty, as a share of the average eigenvalue of X'X. Enough to stop
# correlated signals (trend, price-vs-average and momentum overlap heavily)
# from taking huge offsetting weights; small enough to leave a real signal
# its weight.
RIDGE_PENALTY = 0.10

# Thresholds tried by max_sharpe and search. Scores are normalised so the
# weights' absolute values sum to 1, which keeps a score in [-1, 1].
THETA_GRID = (-0.2, -0.1, 0.0, 0.1, 0.2)

# Random weight vectors tried by search, evaluated in chunks so the
# (T, N, chunk) score tensor stays small in memory.
SEARCH_TRIALS = 2000
SEARCH_CHUNK = 250
SEARCH_SEED = 7

METHODS = ("ridge", "max_sharpe", "search", "equal")


@dataclass(frozen=True)
class Weights:
    method: str
    weights: np.ndarray  # (K,), sum of |w| = 1 unless the method says otherwise
    theta: float
    names: tuple[str, ...]
    train_sharpe: float = float("nan")
    trial_sharpes: np.ndarray | None = field(default=None, repr=False)  # search only

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, (float(v) for v in self.weights)))


@dataclass(frozen=True)
class PortfolioReturns:
    daily: np.ndarray  # (T,) portfolio return per day, NaN where no stock traded
    positions: np.ndarray  # (T, N) 0/1
    turnover: np.ndarray  # (T,) average position changes per stock


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def positions_from_scores(scores: np.ndarray, theta: float, valid: np.ndarray) -> np.ndarray:
    """1 where the score clears theta and the stock has a return that day."""
    return ((scores > theta) & valid).astype(float)


def portfolio_returns(
    positions: np.ndarray,
    returns: np.ndarray,
    previous: np.ndarray | None = None,
    fee_bps: float = FEE_BPS,
) -> PortfolioReturns:
    """Equal-weight portfolio of per-stock long/cash strategies, after costs.

    `previous` is the position held going into the first day (from the prior
    walk-forward period), so a position carried across a refit is not charged
    as a new trade.
    """
    valid = np.isfinite(returns)
    prior = np.zeros(positions.shape[1]) if previous is None else previous
    changes = np.abs(np.diff(np.vstack([prior[None, :], positions]), axis=0))
    per_stock = positions * np.where(valid, returns, 0.0) - changes * fee_bps / 10_000.0
    per_stock = np.where(valid, per_stock, np.nan)
    # A day on which no stock traded has no portfolio return: NaN, not a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        daily = np.nanmean(per_stock, axis=1) if per_stock.size else np.array([])
        turnover = np.nanmean(np.where(valid, changes, np.nan), axis=1)
    return PortfolioReturns(daily, positions, turnover)


def sharpe(daily: np.ndarray) -> float:
    r = daily[np.isfinite(daily)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _batch_sharpes_thetas(
    x: np.ndarray, returns: np.ndarray, weights: np.ndarray, thetas: tuple[float, ...]
) -> np.ndarray:
    """Sharpe after costs for every column of `weights` (K, M) at every threshold.

    Scores for all M weight vectors are one product, X (T,N,K) @ W (K,M),
    computed once and reused for each threshold. Returns shape (len(thetas), M).
    """
    valid = np.isfinite(returns)
    r = np.where(valid, returns, 0.0).astype(np.float32)[:, :, None]
    scores = np.einsum("tnk,km->tnm", x.astype(np.float32), weights.astype(np.float32),
                       optimize=True)
    counts = valid.sum(axis=1)
    usable = counts > 0
    fee = np.float32(FEE_BPS / 10_000.0)
    out = np.full((len(thetas), weights.shape[1]), np.nan)
    for i, theta in enumerate(thetas):
        pos = ((scores > theta) & valid[:, :, None]).astype(np.float32)
        changes = np.abs(np.diff(pos, axis=0, prepend=np.float32(0.0)))
        daily = (pos * r - changes * fee).sum(axis=1, dtype=np.float64)
        daily = daily[usable] / counts[usable, None]
        std = daily.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(std > 0, daily.mean(axis=0) / std * np.sqrt(TRADING_DAYS), np.nan)
    return out


def _batch_sharpes(
    x: np.ndarray, returns: np.ndarray, weights: np.ndarray, theta: float
) -> np.ndarray:
    """Sharpe after costs for every column of `weights` (K, M) at one threshold."""
    return _batch_sharpes_thetas(x, returns, weights, (theta,))[0]


def _normalise(w: np.ndarray) -> np.ndarray:
    total = np.abs(w).sum()
    return w / total if total > 0 else w


def evaluate(x: np.ndarray, returns: np.ndarray, w: Weights,
             previous: np.ndarray | None = None) -> PortfolioReturns:
    scores = x @ w.weights
    positions = positions_from_scores(scores, w.theta, np.isfinite(returns))
    return portfolio_returns(positions, returns, previous)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_ridge(x: np.ndarray, returns: np.ndarray, names: tuple[str, ...],
              penalty: float = RIDGE_PENALTY) -> Weights:
    """w = (X'X + lambda I)^-1 X'y on volatility-scaled next-day returns.

    Centring X and y removes the intercept from the penalty; it is recovered
    as b = mean(y) - mean(x) @ w, and the strategy is long when x @ w + b > 0,
    so theta = -b on the unnormalised scale. Weights and theta are then scaled
    together so sum |w| = 1, which leaves every decision unchanged.
    """
    t, n, k = x.shape
    scale = np.nanstd(returns, axis=0, ddof=1)
    scale = np.where(scale > 0, scale, np.nan)
    y = (returns / scale[None, :]).reshape(-1)
    rows = x.reshape(-1, k)
    keep = np.isfinite(y)
    rows, y = rows[keep], y[keep]

    x_mean, y_mean = rows.mean(axis=0), y.mean()
    xc, yc = rows - x_mean, y - y_mean
    gram = xc.T @ xc
    lam = penalty * np.trace(gram) / k
    w = np.linalg.solve(gram + lam * np.eye(k), xc.T @ yc)
    b = y_mean - x_mean @ w
    total = np.abs(w).sum()
    if total == 0:
        return Weights("ridge", w, 0.0, names)
    weights, theta = w / total, -b / total
    fitted = Weights("ridge", weights, float(theta), names)
    return Weights("ridge", weights, float(theta), names,
                   train_sharpe=sharpe(evaluate(x, returns, fitted).daily))


def signal_returns(x: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """(T, K): each signal traded on its own, position proportional to the signal,
    averaged across stocks."""
    valid = np.isfinite(returns)
    r = np.where(valid, returns, 0.0)
    per = x * r[:, :, None]
    counts = valid.sum(axis=1)
    usable = counts > 0
    out = np.full((x.shape[0], x.shape[2]), np.nan)
    out[usable] = per.sum(axis=1)[usable] / counts[usable, None]
    return out


def _best_theta(x, returns, weights: np.ndarray) -> tuple[float, float]:
    sharpes = [_batch_sharpes(x, returns, weights[:, None], th)[0] for th in THETA_GRID]
    best = int(np.nanargmax(sharpes)) if np.isfinite(sharpes).any() else 2
    return THETA_GRID[best], float(sharpes[best])


def fit_max_sharpe(x: np.ndarray, returns: np.ndarray, names: tuple[str, ...]) -> Weights:
    """Tangency weights w ~ Sigma^-1 mu of the signals' own return streams.

    Solved with np.linalg.solve rather than an explicit inverse. Sigma comes
    from Ledoit-Wolf shrinkage toward a scaled identity, which keeps the solve
    well conditioned when signals are nearly collinear.
    """
    f = signal_returns(x, returns)
    f = f[np.isfinite(f).all(axis=1)]
    mu = f.mean(axis=0)
    sigma = shrink_identity(f).covariance
    w = _normalise(np.linalg.solve(sigma, mu))
    theta, train = _best_theta(x, returns, w)
    return Weights("max_sharpe", w, theta, names, train_sharpe=train)


def fit_search(x: np.ndarray, returns: np.ndarray, names: tuple[str, ...],
               trials: int = SEARCH_TRIALS, seed: int = SEARCH_SEED) -> Weights:
    """Best of `trials` random weight vectors x THETA_GRID by training Sharpe."""
    k = x.shape[2]
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(-1.0, 1.0, size=(k, trials))
    candidates /= np.abs(candidates).sum(axis=0, keepdims=True)
    all_sharpes = []
    best = (-np.inf, None, 0.0)
    for start in range(0, trials, SEARCH_CHUNK):
        block = candidates[:, start:start + SEARCH_CHUNK]
        grid = _batch_sharpes_thetas(x, returns, block, THETA_GRID)  # (thetas, chunk)
        all_sharpes.append(grid.ravel())
        if np.isfinite(grid).any() and np.nanmax(grid) > best[0]:
            ti, i = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
            best = (float(grid[ti, i]), block[:, i].copy(), THETA_GRID[ti])
    score, w, theta = best
    if w is None:
        w = np.full(k, 1.0 / k)
    return Weights("search", w, float(theta), names, train_sharpe=score,
                   trial_sharpes=np.concatenate(all_sharpes))


def fit_equal(x: np.ndarray, returns: np.ndarray, names: tuple[str, ...]) -> Weights:
    k = x.shape[2]
    w = np.full(k, 1.0 / k)
    fitted = Weights("equal", w, 0.0, names)
    return Weights("equal", w, 0.0, names,
                   train_sharpe=sharpe(evaluate(x, returns, fitted).daily))


def fixed(names: tuple[str, ...], weights: dict[str, float], theta: float = 0.0,
          label: str = "fixed") -> Weights:
    """A hand-set rule, e.g. the 50/200 crossover: {"trend": 1.0}."""
    w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    return Weights(label, w, theta, names)


FITTERS = {"ridge": fit_ridge, "max_sharpe": fit_max_sharpe, "search": fit_search,
           "equal": fit_equal}
