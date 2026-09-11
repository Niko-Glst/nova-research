"""Combines all upstream agent outputs into a single investment thesis.

Consumes TechnicalRead, FundamentalRead, SentimentRead, BoardVerdict, and
PodcastSummary (where available) and synthesizes a coherent narrative +
thesis that strategy_agent.py can act on. Research output only — no order
placement happens anywhere downstream of this.

TODO:
- Implement synthesis logic (likely LLM-backed) that reconciles
  disagreements between upstream signals rather than just concatenating them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvestmentThesis:
    symbol: str
    thesis_summary: str
    supporting_points: list[str]
    contradicting_points: list[str]
    conviction: float  # 0.0 - 1.0


def synthesize(
    symbol: str,
    technical: dict,
    fundamental: dict,
    sentiment: dict,
    board_verdict: dict,
) -> InvestmentThesis:
    """Combine upstream analyses into a single InvestmentThesis.

    TODO: implement.
    """
    raise NotImplementedError
