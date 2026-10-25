"""Model versioning, weighted flows, rescoring and the version break in the index."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from agents import breadth_analyst as ba
from agents import flow_analyst as fl
from agents import implied_move, insider_analyst, report_archive, rescore, research_report
from agents import sentiment_analyst, signal_snapshots, ta_analyst
from agents.fa_analyst import FundamentalRead

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _flows() -> fl.FlowRead:
    parts = (
        fl.FlowComponent("issuance", "Net issuance", fl.issuance_score(0.171), "",
                         MappingProxyType({"net_issuance": 0.171}), fl.CONFIRMED_ZERO),
        fl.FlowComponent("systematic_market", "Market",
                         fl.market_score((0.0002, 0.051, 0.165), 0.72), "",
                         MappingProxyType({"return_1m": 0.0002, "return_3m": 0.051,
                                           "return_12m": 0.165, "vol_ratio": 0.72})),
        fl.FlowComponent("systematic_stock", "Stock", fl.stock_score(-0.9, 0.85), "",
                         MappingProxyType({"z": -0.9, "vol_ratio": 0.85}),
                         "vol_estimator: yang_zhang"),
    )
    signal, score = fl.classify(parts)
    return fl.FlowRead(symbol="TEST", signal=signal, score=score, components=parts)


def _breadth() -> ba.BreadthRead:
    """Mid-range, falling 3-month trend: narrowing, so a breadth stock reads bearish."""
    regime = ba.BreadthRegime(ratio=0.2849, low=0.2710, high=0.2976, position=0.5226,
                              ma_turn=0.2803, trend_slope=-0.00088, regime=-1.0,
                              state="trend_narrowing")
    return ba.BreadthRead(symbol="TEST", signal="bearish", score=-1.0, group=ba.BREADTH,
                          sector="Financial Services", market_cap=7e11, regime=regime)


@pytest.fixture
def live_report(monkeypatch):
    monkeypatch.setattr(ta_analyst, "analyze", lambda *a, **k: ta_analyst.TechnicalRead(
        symbol="TEST", indicators={"close": 10.0}, signal="neutral", notes=""))
    monkeypatch.setattr(research_report.fa_analyst, "analyze", lambda *a, **k: FundamentalRead(
        symbol="TEST", metrics={}, signal="overvalued", notes=""))
    monkeypatch.setattr(sentiment_analyst, "analyze", lambda *a, **k: sentiment_analyst.SentimentRead(
        symbol="TEST", reddit_score=0.0, news_score=0.2, sample_size=3, notes="",
        signal="bullish"))
    monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: insider_analyst.InsiderRead(
        symbol="TEST", signal="bearish", lookback_days=180, buy_count=0, sell_count=7,
        buy_value=0.0, sell_value=5e6, net_value=-5e6, senior_buy_count=0,
        senior_sell_count=0))
    monkeypatch.setattr(fl, "analyze", lambda *a, **k: _flows())
    monkeypatch.setattr(ba, "analyze", lambda *a, **k: _breadth())
    monkeypatch.setattr(implied_move, "analyze", lambda *a, **k: implied_move.MarketMove(
        session=datetime(2026, 10, 23).date(), move=-0.0052, vol_prev=19.03, implied=0.012,
        multiple=-0.43, tier=implied_move.NOISE))
    return research_report.analyze("TEST")


class TestModelVersion:
    def test_every_report_carries_the_configured_version(self, live_report):
        assert live_report.model_version == research_report.MODEL_VERSION == "3.0.0"
        assert "Model:      3.0.0" in research_report.format_report(live_report)
        row = signal_snapshots.snapshot_row(live_report, NOW)
        assert row["model_version"] == "3.0.0"

    def test_leg_weights_come_from_config_unchanged(self):
        assert research_report.WEIGHTS == {"fundamental": 1.0, "technical": 0.8, "insider": 0.6,
                                           "sentiment": 0.4, "flows": 0.4, "breadth": 0.5}

    def test_a_2_0_0_run_rescores_under_3_0_0(self, live_report):
        """2.0.0 rows stored every input, so the new weight and rules apply to them."""
        row = {**signal_snapshots.snapshot_row(live_report, NOW), "model_version": "2.0.0"}
        result = rescore.rescore_row(row)
        assert result.rescored and result.original_model == "2.0.0"
        assert result.model_version == "3.0.0"


class TestWeightedFlows:
    def test_missing_part_is_reweighted_not_zero(self):
        weights = {"a": 1.0, "b": 3.0}
        assert fl.weighted_score({"a": 1.0, "b": None}, weights) == 1.0
        assert fl.weighted_score({"a": 1.0, "b": -1.0}, weights) == pytest.approx(-0.5)
        assert math.isnan(fl.weighted_score({"a": None}, weights))

    def test_score_always_in_range(self):
        assert -1.0 <= fl.weighted_score({"issuance": -1.0, "systematic_market": 1.0,
                                          "systematic_stock": 0.3}) <= 1.0


class TestRescore:
    def test_current_model_run_reproduces_its_score_exactly(self, live_report):
        row = signal_snapshots.snapshot_row(live_report, NOW)
        result = rescore.rescore_row(row)
        assert result.rescored
        assert result.score == pytest.approx(live_report.score, abs=1e-12)
        assert result.verdict == live_report.verdict
        assert result.flows_signal == live_report.flows.signal
        assert rescore.breadth_stance(row) == live_report.breadth.signal == "bearish"

    def test_breadth_rule_from_stored_numbers(self):
        """The stored summary numbers give the same regime as the live rule."""
        assert rescore.regime_from_inputs(0.272, 0.10, 0.271, -0.001) == ba.ZONE_TURN_SCORE
        assert rescore.regime_from_inputs(0.272, 0.10, 0.275, +0.001) == ba.ZONE_WAITING_SCORE
        assert rescore.regime_from_inputs(0.296, 0.95, 0.297, +0.001) == -ba.ZONE_TURN_SCORE
        assert rescore.regime_from_inputs(0.296, 0.95, 0.290, -0.001) == -ba.ZONE_WAITING_SCORE
        assert rescore.regime_from_inputs(0.285, 0.50, 0.290, +0.001) == ba.TREND_SCORE
        assert rescore.regime_from_inputs(0.285, 0.50, 0.280, -0.001) == -ba.TREND_SCORE

    def test_waiting_in_a_zone_rescores_as_no_call(self):
        row = {"breadth_group": ba.BREADTH, "breadth_ratio": "0.272", "breadth_position": "0.1",
               "breadth_ma_turn": "0.275", "breadth_trend_slope": "0.001"}  # no turn yet
        assert rescore.breadth_stance(row) == "unknown"

    def test_breadth_without_inputs_is_unknown(self):
        assert rescore.breadth_stance({"breadth_group": "breadth"}) == "unknown"
        assert rescore.breadth_stance({"breadth_ratio": "0.28", "breadth_position": "0.5",
                                       "breadth_ma_turn": "0.28",
                                       "breadth_trend_slope": "0.001"}) == "unknown"

    def test_leader_reads_the_opposite_on_rescore(self):
        row = {"breadth_group": ba.LEADER, "breadth_ratio": "0.285", "breadth_position": "0.5",
               "breadth_ma_turn": "0.28", "breadth_trend_slope": "-0.001"}
        assert rescore.breadth_stance(row) == "bullish"

    def test_old_runs_lack_inputs_and_are_not_written(self, tmp_path):
        old = {"symbol": "IBRX", "recorded_at": "2026-10-17T19:53:52+00:00",
               "technical_signal": "bullish", "flows_signal": ""}
        result = rescore.rescore_row(old)
        assert not result.rescored and "net_issuance" in result.missing
        assert signal_snapshots.record_rescores([result], tmp_path / "r.csv", NOW) == 0

    def test_history_rescore_is_idempotent_and_never_edits(self, live_report, tmp_path,
                                                            monkeypatch, capsys):
        import history

        snapshots, rescored = tmp_path / "signals.csv", tmp_path / "rescored.csv"
        signal_snapshots.record(live_report, snapshots, NOW)
        monkeypatch.setattr(signal_snapshots, "SNAPSHOT_PATH", snapshots)
        monkeypatch.setattr(signal_snapshots, "RESCORED_PATH", rescored)
        monkeypatch.setattr(signal_snapshots.snapshot_store, "STORE_DIR", tmp_path / "long")
        original = snapshots.read_text(encoding="utf-8")
        assert history.main(["rescore"]) == 0
        assert history.main(["rescore"]) == 0
        assert snapshots.read_text(encoding="utf-8") == original
        assert len(signal_snapshots.load_rows(rescored)) == 1
        assert "1 already rescored" in capsys.readouterr().out


class TestVersionBreak:
    def _run(self, minute, version, score):
        return report_archive.ArchivedRun(
            symbol="IBRX", generated=datetime(2026, 10, 22, 12, minute),
            path=report_archive.Path("/r/archive/IBRX.html"), verdict="mixed", score=score,
            confidence=100, model_version=version)

    def test_no_change_is_computed_across_versions(self):
        root = report_archive.Path("/r")
        newer, older = self._run(30, "2.0.0", -0.10), self._run(0, "1.1.0", -0.37)
        row = report_archive._row(newer, older, root)
        assert "version break" in row and "1.1.0 &rarr; 2.0.0" in row
        assert "+0.27" not in row  # the difference is never shown
        assert "lacks the inputs" in row

    def test_change_within_a_version(self):
        root = report_archive.Path("/r")
        row = report_archive._row(self._run(30, "2.0.0", -0.10), self._run(0, "2.0.0", -0.37),
                                  root)
        assert "+0.27" in row and "version break" not in row

    def test_unversioned_reports_are_inferred(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "X_2026-10-17_1953.html").write_text(
            '<span class="verdict">MIXED</span> score -0.14 confidence 100% Market flows',
            encoding="utf-8")
        (archive / "X_2026-10-08_1000.html").write_text(
            '<span class="verdict">MIXED</span> score -0.10 confidence 75%', encoding="utf-8")
        runs = report_archive.list_runs(tmp_path)["X"]
        assert [r.model_version for r in runs] == [report_archive.INFERRED_FLOWS,
                                                   report_archive.INFERRED_BASE]
