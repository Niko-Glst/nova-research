"""Risk and position size for one stock, shown beside the verdict and never scored.

Feeds agents/risk_manager.py (the gate) and agents/allocation_decision.py (the
sizing methods) with returns from correlations.load_returns: cached closes,
inner-joined on the dates both markets traded, in local currencies.

Two stated simplifications:

- The project stores no holdings, so the listing's local index stands in for the
  book. Correlation "to the book" therefore reads as correlation to the market,
  which is what a diversified holder already owns.
- Conviction is the report's score when it is positive and zero otherwise. The
  size at full conviction is shown next to it, so the size the volatility allows
  stays visible for a stock the score does not favour.

Sizes are percentages of equity. Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from agents import allocation_decision, benchmarks, risk_manager
from config.settings import load_scoring_config
from correlations import load_returns

LOOKBACK_YEARS = int(load_scoring_config()["risk_read"]["lookback_years"])

# Sizes are read as percentages, so the equity only has to be a round number.
SIZING_EQUITY = 100.0


@dataclass(frozen=True)
class SizeLine:
    """One sizing method at the report's score and at full conviction."""

    method: str
    at_score: allocation_decision.AllocationDecision
    at_full_conviction: allocation_decision.AllocationDecision


@dataclass(frozen=True)
class PeerCorrelation:
    peer: str
    correlation: float
    observations: int


@dataclass(frozen=True)
class RiskRead:
    symbol: str
    benchmark: str
    start: str  # first date of the window
    conviction: float  # 0.0 .. 1.0
    assessment: risk_manager.RiskAssessment
    sizes: tuple[SizeLine, ...]
    peers: tuple[PeerCorrelation, ...] = ()
    peers_unavailable: tuple[str, ...] = ()


def analyze(
    symbol: str,
    score: float,
    peers: Sequence[str] = (),
    now: datetime | None = None,
    **load_options,
) -> RiskRead:
    """The gate, the sizes and the correlation with each peer.

    `load_options` go to correlations.load_returns (the fetcher and the cache
    directory in tests). A peer that cannot be loaded is named, not fatal: peer
    lists can hold delisted tickers.
    """
    symbol = symbol.strip().upper()
    benchmark = benchmarks.benchmark_for(symbol)
    if benchmark is None:
        raise ValueError(f"No local index is configured for {symbol}.")
    start = (pd.Timestamp(now or datetime.now()) - pd.DateOffset(years=LOOKBACK_YEARS))
    start_text = start.date().isoformat()

    def pair(other: str) -> pd.DataFrame:
        data = load_returns([symbol, other], base=None, start=start_text, now=now, **load_options)
        return data.returns

    logs = pair(benchmark)
    simple = np.expm1(logs)  # load_returns gives log returns; the gate compounds simple ones
    assessment = risk_manager.assess(symbol, simple[symbol], simple[[benchmark.upper()]])

    conviction = max(0.0, min(1.0, float(score)))
    inputs = asdict(assessment)
    sizes = tuple(
        SizeLine(
            method.value,
            allocation_decision.size_position(symbol, conviction, inputs, SIZING_EQUITY,
                                              method=method),
            allocation_decision.size_position(symbol, 1.0, inputs, SIZING_EQUITY, method=method),
        )
        for method in allocation_decision.SizingMethod
    )

    readings: list[PeerCorrelation] = []
    missing: list[str] = []
    for peer in peers:
        peer = peer.strip().upper()
        if peer == symbol:
            continue
        try:
            returns = pair(peer)
        except Exception:
            missing.append(peer)
            continue
        readings.append(PeerCorrelation(peer, float(returns[symbol].corr(returns[peer])),
                                        len(returns)))

    return RiskRead(symbol=symbol, benchmark=benchmark, start=start_text, conviction=conviction,
                    assessment=assessment, sizes=sizes, peers=tuple(readings),
                    peers_unavailable=tuple(missing))
