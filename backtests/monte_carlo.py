"""Monte Carlo simulation of strategy returns.

Resamples historical (or backtested) returns to estimate a distribution of
outcomes for a strategy, rather than relying on a single historical path.
Simulation only.

TODO:
- Implement a resampling method (e.g. bootstrap of daily returns, or block
  bootstrap to preserve autocorrelation).
- Implement summary statistics over the simulated distribution (percentiles
  of terminal return, probability of drawdown beyond X%, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    symbol: str
    num_simulations: int
    return_percentiles: dict[str, float]  # e.g. {"p5": ..., "p50": ..., "p95": ...}
    probability_of_loss: float


def resample_returns(returns: np.ndarray, num_simulations: int, horizon_days: int) -> np.ndarray:
    """Bootstrap-resample a returns series into (num_simulations, horizon_days) paths.

    TODO: implement.
    """
    raise NotImplementedError


def run_monte_carlo(
    symbol: str, returns: np.ndarray, num_simulations: int = 1000, horizon_days: int = 252
) -> MonteCarloResult:
    """Run a Monte Carlo simulation over a strategy's historical returns.

    TODO: implement, composing resample_returns + percentile/probability summary.
    """
    raise NotImplementedError
