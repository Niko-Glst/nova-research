"""Correlation and risk-adjusted return checks.

Reviews proposed strategy signals against the rest of a (simulated) portfolio
for correlation and risk-adjusted return before allocation_decision.py sizes
anything. Purely analytical — no order placement.

TODO:
- Implement correlation checks against existing/simulated portfolio holdings.
- Implement risk-adjusted return metrics (e.g. Sharpe/Sortino) gating.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskAssessment:
    symbol: str
    correlation_to_portfolio: float
    sharpe_ratio: float
    sortino_ratio: float
    approved: bool
    reason: str


def compute_correlation(
    candidate_returns: pd.Series, portfolio_returns: pd.DataFrame
) -> float:
    """Compute correlation of a candidate position's returns to the existing portfolio.

    TODO: implement.
    """
    raise NotImplementedError


def compute_risk_adjusted_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute Sharpe/Sortino (or similar) for a returns series.

    TODO: implement.
    """
    raise NotImplementedError


def assess(
    symbol: str, candidate_returns: pd.Series, portfolio_returns: pd.DataFrame
) -> RiskAssessment:
    """Run the full risk assessment for a candidate position.

    TODO: implement, composing compute_correlation + compute_risk_adjusted_metrics
    into an approve/reject decision with a stated reason.
    """
    raise NotImplementedError
