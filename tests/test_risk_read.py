"""The risk read: gate and sizes against the local index, every wired-in rule, and the report."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from agents import allocation_decision, benchmarks, html_report, research_report, risk_read
from correlations import data as correlation_data

NOW = datetime(2026, 11, 2, 12, 0)


def _frame(seed: int, drift: float, market: np.ndarray | None = None, days: int = 1500):
    """A price frame ending at NOW; with `market`, its returns follow that series."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.012, days)
    returns = noise + drift if market is None else 0.8 * market + 0.5 * noise + drift
    index = pd.bdate_range(end=NOW.date(), periods=days)
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close,
                         "Volume": 1000.0}, index=index), returns


@pytest.fixture
def options(tmp_path):
    """Offline prices: the index, a stock that follows it, and a peer. Others do not exist."""
    index_frame, market = _frame(1, 0.0004)
    frames = {"SPY": index_frame, "GOOD": _frame(2, 0.0009, market)[0],
              "PEER": _frame(3, 0.0003, market)[0], "BAD": _frame(4, -0.0015)[0]}

    def fetcher(symbol, period):
        if symbol not in frames:
            raise ValueError(f"no data for {symbol}")
        return frames[symbol]

    return {"fetcher": fetcher, "cache_dir": tmp_path}


class TestRiskRead:
    def test_the_gate_reads_against_the_local_index(self, options):
        read = risk_read.analyze("GOOD", 0.4, now=NOW, **options)
        assert read.benchmark == "SPY" and read.assessment.approved
        assert read.assessment.correlation_to_portfolio > 0.5
        assert read.assessment.observations > 1000

    def test_every_sizing_method_at_the_score_and_at_full_conviction(self, options):
        read = risk_read.analyze("GOOD", 0.4, now=NOW, **options)
        assert {s.method for s in read.sizes} == {m.value for m in allocation_decision.SizingMethod}
        for size in read.sizes:
            assert size.at_score.position_size_pct <= size.at_full_conviction.position_size_pct
        assert read.conviction == pytest.approx(0.4)

    def test_a_negative_score_means_no_conviction_but_shows_what_volatility_allows(self, options):
        read = risk_read.analyze("GOOD", -0.3, now=NOW, **options)
        assert read.conviction == 0.0
        target = next(s for s in read.sizes if s.method == "volatility_target")
        assert target.at_score.position_size_pct == 0.0
        assert target.at_full_conviction.position_size_pct > 0.0

    def test_a_failed_gate_sizes_nothing(self, options):
        read = risk_read.analyze("BAD", 0.5, now=NOW, **options)
        assert not read.assessment.approved and "Sharpe" in read.assessment.reason
        assert all(s.at_full_conviction.position_size_pct == 0.0 for s in read.sizes)

    def test_a_dead_peer_is_named_not_fatal(self, options):
        read = risk_read.analyze("GOOD", 0.4, peers=["PEER", "GOOD", "VMW"], now=NOW, **options)
        assert [p.peer for p in read.peers] == ["PEER"]
        assert read.peers[0].correlation > 0.3
        assert read.peers_unavailable == ("VMW",)

    def test_an_unconfigured_listing_says_so(self, options):
        with pytest.raises(ValueError, match="local index"):
            risk_read.analyze("XYZ.OL", 0.4, now=NOW, **options)

    @pytest.mark.parametrize("suffix", sorted(benchmarks.BENCHMARKS))
    def test_every_benchmark_and_listing_is_known_to_the_returns_loader(self, suffix):
        """load_returns raises on an unknown listing, which would blank the section."""
        config = correlation_data.load_correlation_config()
        correlation_data.listing_of(benchmarks.BENCHMARKS[suffix], config)
        correlation_data.listing_of(f"ABC{suffix}", config)


class TestJsonOutput:
    def test_read_only_mappings_arrays_and_nan_all_serialize(self, options):
        """`--json` crashed on the flow components' read-only mappings."""
        import json
        from types import MappingProxyType

        import research

        report = research_report.ResearchReport(
            symbol="GOOD", verdict="positive", score=float("nan"), confidence=1.0,
            risk=risk_read.analyze("GOOD", 0.4, now=NOW, **options),
            simulated_returns=np.array([1.0, 2.0]),
            buy_hold_equity=pd.Series([1.0, 1.1]),
            errors=MappingProxyType({"flows": "x"}))
        loaded = json.loads(json.dumps(research._to_jsonable(report), allow_nan=False, default=str))
        assert loaded["score"] is None and loaded["simulated_returns"] == [1.0, 2.0]
        assert loaded["errors"] == {"flows": "x"} and loaded["risk"]["benchmark"] == "SPY"


class TestInTheReport:
    @pytest.fixture
    def report(self, options):
        return research_report.ResearchReport(
            symbol="GOOD", verdict="positive", score=0.4, confidence=1.0,
            risk=risk_read.analyze("GOOD", 0.4, peers=["PEER", "VMW"], now=NOW, **options),
            model_version="4.0.0")

    def test_text_report(self, report):
        text = research_report.format_report(report)
        assert "RISK AND SIZE" in text and "volatility_target" in text
        assert "SPY stands in for the book" in text
        assert "PEER" in text and "VMW" in text

    def test_html_report(self, report):
        page = html_report.render_report(report, "2026-11-02")
        assert "Risk and position size" in page and "volatility target" in page
        assert "never scored" in page and "PEER" in page and "VMW" in page
