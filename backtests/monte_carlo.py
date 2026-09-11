"""Monte Carlo simulation of strategy returns.

A backtest gives you one path: the one that happened. That single number hides
how much of the outcome was luck. Resampling the same returns into thousands of
alternative orderings shows the distribution the strategy could plausibly have
produced, which is the honest way to read a backtest result.

Two resampling methods, and the choice between them matters:

- **IID bootstrap** draws days independently. Simple, but it destroys
  autocorrelation and volatility clustering -- real markets have both, so this
  understates the tails.
- **Block bootstrap** draws contiguous runs of days, preserving short-range
  structure within each block. This is the default for that reason.

Simulation only; nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252

# Block length for the block bootstrap, in trading days. Roughly a month:
# long enough to carry volatility clustering, short enough that the resampled
# paths are not just copies of the original.
DEFAULT_BLOCK_SIZE = 21

# Percentiles reported by default.
DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class MonteCarloResult:
    symbol: str
    num_simulations: int
    return_percentiles: dict[str, float]  # e.g. {"p5": ..., "p50": ..., "p95": ...}
    probability_of_loss: float
    # Beyond the original four fields:
    horizon_days: int = 0
    method: str = "block"
    mean_return_pct: float = 0.0
    median_return_pct: float = 0.0
    probability_of_drawdown_20pct: float = 0.0
    max_drawdown_percentiles: dict[str, float] = field(default_factory=dict)
    value_at_risk_5pct: float = 0.0
    conditional_var_5pct: float = 0.0
    observed_return_pct: float | None = None

    @property
    def observed_percentile(self) -> float | None:
        """Where the realised result sits in the simulated distribution.

        A backtest that lands at the 95th percentile of its own resampled
        distribution was lucky, not skilful -- this is the number that says so.
        """
        return self._observed_percentile

    _observed_percentile: float | None = None

    def summary(self) -> str:
        lines = [
            f"Monte Carlo: {self.symbol}",
            f"  {self.num_simulations:,} simulations, {self.horizon_days}-day horizon, "
            f"{self.method} bootstrap",
            "",
            "  Terminal return distribution:",
        ]
        for name in sorted(self.return_percentiles, key=lambda k: int(k[1:])):
            lines.append(f"    {name:>5}: {self.return_percentiles[name]:>8.2f}%")
        lines += [
            "",
            f"  mean:                {self.mean_return_pct:>8.2f}%",
            f"  probability of loss: {self.probability_of_loss:>8.1%}",
            f"  P(drawdown > 20%):   {self.probability_of_drawdown_20pct:>8.1%}",
            f"  VaR (5%):            {self.value_at_risk_5pct:>8.2f}%",
            f"  CVaR (5%):           {self.conditional_var_5pct:>8.2f}%  "
            f"(mean of the worst 5%)",
        ]
        if self.observed_return_pct is not None and self._observed_percentile is not None:
            lines += [
                "",
                f"  observed backtest:   {self.observed_return_pct:>8.2f}%  "
                f"(percentile {self._observed_percentile:.0f} of this distribution)",
            ]
            if self._observed_percentile > 90:
                lines.append(
                    "  The realised path sits in the top decile of what resampling "
                    "produces: treat it as a favourable draw, not a reliable expectation."
                )
        return "\n".join(lines)


def resample_returns(
    returns: np.ndarray,
    num_simulations: int,
    horizon_days: int,
    method: str = "block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap a returns series into (num_simulations, horizon_days) paths.

    `method` is "block" (preserves short-range autocorrelation) or "iid".
    Pass `seed` for a reproducible draw.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[~np.isnan(returns)]

    if returns.size < 2:
        raise ValueError(
            f"Need at least 2 return observations to resample, got {returns.size}."
        )
    if num_simulations < 1 or horizon_days < 1:
        raise ValueError("num_simulations and horizon_days must both be positive.")

    rng = np.random.default_rng(seed)

    if method == "iid":
        indices = rng.integers(0, returns.size, size=(num_simulations, horizon_days))
        return returns[indices]

    if method != "block":
        raise ValueError(f"Unknown method {method!r}; expected 'block' or 'iid'.")

    # Block bootstrap: draw ceil(horizon/block) starting points per simulation,
    # lay their blocks end to end, then trim to the horizon.
    effective_block = min(block_size, returns.size)
    num_blocks = int(np.ceil(horizon_days / effective_block))
    max_start = returns.size - effective_block + 1

    starts = rng.integers(0, max_start, size=(num_simulations, num_blocks))
    # offsets broadcast each start into a contiguous run of indices.
    offsets = np.arange(effective_block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(
        num_simulations, num_blocks * effective_block
    )
    return returns[indices[:, :horizon_days]]


def _path_statistics(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Terminal return and maximum drawdown for each simulated path."""
    equity = np.cumprod(1.0 + paths, axis=1)
    terminal = (equity[:, -1] - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / running_max - 1.0
    max_drawdown = drawdowns.min(axis=1) * 100.0

    return terminal, max_drawdown


def run_monte_carlo(
    symbol: str,
    returns: np.ndarray,
    num_simulations: int = 1000,
    horizon_days: int = TRADING_DAYS,
    method: str = "block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    observed_return_pct: float | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Simulate a distribution of outcomes from a strategy's realised returns.

    Pass `observed_return_pct` (the backtest's actual result) to find out where
    that result sits in the distribution -- the single most useful diagnostic
    here, because it separates a good strategy from a lucky one.
    """
    paths = resample_returns(
        returns,
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        method=method,
        block_size=block_size,
        seed=seed,
    )
    terminal, max_drawdown = _path_statistics(paths)

    return_percentiles = {
        f"p{p}": float(np.percentile(terminal, p)) for p in percentiles
    }
    drawdown_percentiles = {
        f"p{p}": float(np.percentile(max_drawdown, p)) for p in percentiles
    }

    var_5 = float(np.percentile(terminal, 5))
    tail = terminal[terminal <= var_5]
    cvar_5 = float(tail.mean()) if tail.size else var_5

    observed_percentile = None
    if observed_return_pct is not None:
        observed_percentile = float(
            (terminal < observed_return_pct).sum() / terminal.size * 100.0
        )

    return MonteCarloResult(
        symbol=symbol,
        num_simulations=num_simulations,
        return_percentiles=return_percentiles,
        probability_of_loss=float((terminal < 0).mean()),
        horizon_days=horizon_days,
        method=method,
        mean_return_pct=float(terminal.mean()),
        median_return_pct=float(np.median(terminal)),
        probability_of_drawdown_20pct=float((max_drawdown <= -20.0).mean()),
        max_drawdown_percentiles=drawdown_percentiles,
        value_at_risk_5pct=var_5,
        conditional_var_5pct=cvar_5,
        observed_return_pct=observed_return_pct,
        _observed_percentile=observed_percentile,
    )
