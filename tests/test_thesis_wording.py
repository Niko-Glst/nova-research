"""Thesis and flows wording must agree with the numbers they describe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents import flow_analyst as fl
from agents import thesis as thesis_module
from agents.fa_analyst import FundamentalRead, PeerComparison


def _peers(forward_pe_rel: float) -> PeerComparison:
    return PeerComparison(niche="Internet Content & Information", peers_used=["GOOGL", "NFLX"],
                          is_sector_fallback=False, symbol_metrics={}, peer_medians={},
                          relative={"forward_pe": forward_pe_rel}, verdict="expensive", notes="")


def _valuation_line(rel: float) -> str:
    fundamental = FundamentalRead(symbol="META", metrics={}, signal="fair", notes="",
                                  peer_comparison=_peers(rel))
    thesis = thesis_module.build_thesis("META", fundamental, None, None, None)
    return next(c.current for c in thesis.bull_case if c.metric == "forward_pe")


@pytest.mark.parametrize(("rel", "text"), [
    (0.85, "at or below the group"),
    (1.0, "at or below the group"),
    (1.38, "a 38% premium, within the 1.5x this condition allows"),  # META, 2026-10-25
])
def test_valuation_wording_matches_the_ratio(rel, text):
    assert text in _valuation_line(rel)


def _ohlc(vol_short: float, vol_long: float, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moves = np.concatenate([rng.normal(0, vol_long, 260), rng.normal(0, vol_short, 21)])
    close = 100 * np.exp(np.cumsum(moves))
    days = pd.bdate_range("2025-01-02", periods=len(close))
    return pd.DataFrame({"Open": close, "High": close * 1.002, "Low": close * 0.998,
                         "Close": close}, index=days)


def test_continuous_volatility_wording_follows_z():
    part = fl.stock_volatility_component(_ohlc(0.008, 0.02), "TEST", scoring="continuous")
    assert part.values["z"] < -1 and part.score == pytest.approx(1.0)
    assert "below its norm by" in part.reason and "within the band" not in part.reason
    up = fl.stock_volatility_component(_ohlc(0.04, 0.015), "TEST", scoring="continuous")
    assert "above its norm by" in up.reason and up.score < 0
