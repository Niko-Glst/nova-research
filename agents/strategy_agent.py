"""Entry/exit strategy logic.

Turns an InvestmentThesis into concrete, backtestable entry/exit signals.
Produces signal arrays consumable by backtests/backtest_engine.py (vectorbt) —
this module never places a real order; it only emits research signals.

TODO:
- Implement rules/model that map a thesis + price history into entries/exits
  (e.g. boolean pandas Series aligned to a price index).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySignals:
    symbol: str
    entries: pd.Series  # boolean series aligned to price index
    exits: pd.Series  # boolean series aligned to price index
    rationale: str


def generate_signals(symbol: str, thesis: dict, price_history: pd.DataFrame) -> StrategySignals:
    """Generate entry/exit signals for a symbol given its thesis and price history.

    TODO: implement.
    """
    raise NotImplementedError
