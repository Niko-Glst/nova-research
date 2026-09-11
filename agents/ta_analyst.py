"""Technical analysis on price data.

Computes technical indicators (via pandas-ta) on OHLCV data pulled through
yfinance and produces a structured technical read for a given symbol.

TODO:
- Implement OHLCV fetch (respecting rate limits — see config/allowed_sources.yaml).
- Implement indicator computation (e.g. moving averages, RSI, MACD via pandas-ta).
- Implement a structured signal summary for downstream agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TechnicalRead:
    symbol: str
    indicators: dict[str, float]
    signal: str  # e.g. "bullish" / "bearish" / "neutral"
    notes: str


def fetch_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV history for a symbol via yfinance, with rate limiting.

    TODO: implement, including backoff/sleep between repeated calls.
    """
    raise NotImplementedError


def compute_indicators(price_history: pd.DataFrame) -> dict[str, float]:
    """Compute technical indicators for the given price history via pandas-ta.

    TODO: implement.
    """
    raise NotImplementedError


def analyze(symbol: str) -> TechnicalRead:
    """Run the full TA pipeline for a symbol.

    TODO: implement, composing fetch_price_history + compute_indicators.
    """
    raise NotImplementedError
