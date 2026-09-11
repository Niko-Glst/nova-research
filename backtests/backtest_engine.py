"""Historical backtest engine, built on vectorbt.

Takes StrategySignals (entries/exits) plus price history and runs a vectorized
historical simulation. Simulation only — no connection to any live/paper
broker order path.

TODO:
- Implement vectorbt Portfolio construction from entries/exits + price data.
- Implement performance-metric extraction (return, drawdown, Sharpe, etc.)
  into a plain-dict/dataclass result so downstream code doesn't need to know
  about vectorbt's internal types.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int


def run_backtest(
    price_history: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    fees_bps: float = 0.0,
) -> BacktestResult:
    """Run a vectorbt-based historical backtest for a single symbol.

    TODO: implement using vectorbt.Portfolio.from_signals, then extract a
    BacktestResult from portfolio.stats().
    """
    raise NotImplementedError
