"""Final position sizing, including transaction costs.

Takes an approved RiskAssessment + StrategySignals and produces a final,
simulated position size. This module NEVER places a real order — it only
produces a research-grade sizing recommendation for paper/simulation use.

TODO:
- Implement sizing logic (e.g. fixed-fractional, volatility-targeted, Kelly-ish)
  that accounts for estimated transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationDecision:
    symbol: str
    position_size_pct: float  # percent of simulated portfolio equity
    estimated_transaction_cost: float
    rationale: str


def size_position(
    symbol: str,
    conviction: float,
    risk_assessment: dict,
    portfolio_equity: float,
    estimated_cost_bps: float,
) -> AllocationDecision:
    """Compute a final simulated position size for a symbol.

    TODO: implement. Must never place an actual order — this is a research
    sizing recommendation only.
    """
    raise NotImplementedError
