"""Fundamental analysis on company financials.

Pulls fundamental data (e.g. via yfinance) and produces a structured
fundamental read for a given symbol (valuation, growth, profitability).

TODO:
- Implement fundamentals fetch (respecting rate limits).
- Implement derived ratio computation (P/E, revenue growth, margins, etc.).
- Implement a structured signal summary for downstream agents.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FundamentalRead:
    symbol: str
    metrics: dict[str, float]
    signal: str  # e.g. "undervalued" / "overvalued" / "fair"
    notes: str


def fetch_fundamentals(symbol: str) -> dict[str, float]:
    """Fetch raw fundamental data for a symbol, with rate limiting.

    TODO: implement.
    """
    raise NotImplementedError


def compute_ratios(fundamentals: dict[str, float]) -> dict[str, float]:
    """Derive valuation/growth/profitability ratios from raw fundamentals.

    TODO: implement.
    """
    raise NotImplementedError


def analyze(symbol: str) -> FundamentalRead:
    """Run the full FA pipeline for a symbol.

    TODO: implement, composing fetch_fundamentals + compute_ratios.
    """
    raise NotImplementedError
