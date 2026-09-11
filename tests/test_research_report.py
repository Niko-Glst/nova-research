"""Tests for the orchestrator's weighting and verdict logic.

Built entirely from hand-made analyst reads, so the combination rules are tested
independently of whether any data provider is reachable.
"""

from __future__ import annotations

import pytest

from agents import insider_analyst, research_report, sentiment_analyst, ta_analyst
from agents.fa_analyst import FundamentalRead
from agents.research_report import ResearchReport


def _technical(signal: str) -> ta_analyst.TechnicalRead:
    return ta_analyst.TechnicalRead(
        symbol="TEST", indicators={"close": 100.0}, signal=signal, notes=""
    )


def _fundamental(signal: str) -> FundamentalRead:
    return FundamentalRead(symbol="TEST", metrics={}, signal=signal, notes="")


def _sentiment(signal: str) -> sentiment_analyst.SentimentRead:
    return sentiment_analyst.SentimentRead(
        symbol="TEST", reddit_score=0.0, news_score=0.0, sample_size=1,
        notes="", signal=signal,
    )


def _insider(signal: str) -> insider_analyst.InsiderRead:
    return insider_analyst.InsiderRead(
        symbol="TEST", signal=signal, lookback_days=180, buy_count=0, sell_count=0,
        buy_value=0.0, sell_value=0.0, net_value=0.0, senior_buy_count=0,
        senior_sell_count=0,
    )


class TestStanceValues:
    @pytest.mark.parametrize("stance", ["bullish", "undervalued"])
    def test_positive_stances(self, stance):
        assert research_report._stance_value(stance) == 1.0

    @pytest.mark.parametrize("stance", ["bearish", "overvalued"])
    def test_negative_stances(self, stance):
        assert research_report._stance_value(stance) == -1.0

    @pytest.mark.parametrize("stance", ["neutral", "fair"])
    def test_neutral_stances(self, stance):
        assert research_report._stance_value(stance) == 0.0

    def test_unknown_carries_no_direction(self):
        """'unknown' means no data, which must not be scored as neutral."""
        assert research_report._stance_value("unknown") is None
        assert research_report._stance_value(None) is None


class TestWeightedScoring:
    def _run(self, monkeypatch, technical, fundamental, sentiment, insider):
        monkeypatch.setattr(ta_analyst, "analyze", lambda *a, **k: _technical(technical))
        monkeypatch.setattr(
            research_report.fa_analyst, "analyze", lambda *a, **k: _fundamental(fundamental)
        )
        monkeypatch.setattr(
            sentiment_analyst, "analyze", lambda *a, **k: _sentiment(sentiment)
        )
        monkeypatch.setattr(
            insider_analyst, "analyze", lambda *a, **k: _insider(insider)
        )
        return research_report.analyze("TEST")

    def test_unanimous_bullish(self, monkeypatch):
        report = self._run(monkeypatch, "bullish", "undervalued", "bullish", "bullish")
        assert report.verdict == "positive"
        assert report.score == pytest.approx(1.0)
        assert report.confidence == pytest.approx(1.0)

    def test_unanimous_bearish(self, monkeypatch):
        report = self._run(monkeypatch, "bearish", "overvalued", "bearish", "bearish")
        assert report.verdict == "negative"
        assert report.score == pytest.approx(-1.0)

    def test_known_weighting_case(self, monkeypatch):
        """FA +1.0, TA +0.8, insider -0.6, sentiment 0 -> +1.2/2.8."""
        report = self._run(monkeypatch, "bullish", "undervalued", "neutral", "bearish")
        assert report.score == pytest.approx(1.2 / 2.8, abs=1e-6)
        assert report.verdict == "positive"

    def test_fundamentals_outweigh_sentiment(self, monkeypatch):
        report = self._run(monkeypatch, "neutral", "undervalued", "bearish", "neutral")
        assert report.score > 0

    def test_conflicting_signals_read_mixed(self, monkeypatch):
        report = self._run(monkeypatch, "bullish", "overvalued", "neutral", "neutral")
        assert report.verdict == "mixed"


class TestMissingAnalysts:
    def test_unknown_stances_lower_confidence(self, monkeypatch):
        monkeypatch.setattr(ta_analyst, "analyze", lambda *a, **k: _technical("bullish"))
        monkeypatch.setattr(
            research_report.fa_analyst, "analyze", lambda *a, **k: _fundamental("undervalued")
        )
        monkeypatch.setattr(
            sentiment_analyst, "analyze", lambda *a, **k: _sentiment("unknown")
        )
        monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: _insider("unknown"))

        report = research_report.analyze("TEST")
        # Only FA (1.0) and TA (0.8) reported, out of 2.8 total weight.
        assert report.confidence == pytest.approx(1.8 / 2.8, abs=1e-6)
        assert report.score == pytest.approx(1.0)

    def test_a_failing_analyst_is_recorded_not_raised(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(ta_analyst, "analyze", _boom)
        monkeypatch.setattr(
            research_report.fa_analyst, "analyze", lambda *a, **k: _fundamental("undervalued")
        )
        monkeypatch.setattr(
            sentiment_analyst, "analyze", lambda *a, **k: _sentiment("neutral")
        )
        monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: _insider("neutral"))

        report = research_report.analyze("TEST")
        assert "technical" in report.errors
        assert "provider down" in report.errors["technical"]
        assert report.verdict == "positive"  # the other three still counted

    def test_all_analysts_failing_is_inconclusive(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("offline")

        for module in (ta_analyst, sentiment_analyst, insider_analyst):
            monkeypatch.setattr(module, "analyze", _boom)
        monkeypatch.setattr(research_report.fa_analyst, "analyze", _boom)

        report = research_report.analyze("TEST")
        assert report.verdict == "inconclusive"
        assert report.confidence == 0.0
        assert len(report.errors) == 4


class TestValidationAndFormatting:
    def test_blank_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            research_report.analyze("   ")

    def test_format_report_handles_a_bare_report(self):
        text = research_report.format_report(
            ResearchReport(symbol="TEST", verdict="inconclusive", score=0.0, confidence=0.0)
        )
        assert "TEST" in text
        assert "not financial advice" in text

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(5.27e12, "$5.27T"), (1.17e9, "$1.17B"), (250_000.0, "$250.00K"), (99.0, "$99.00")],
    )
    def test_money_formatting(self, value, expected):
        assert research_report._format_money(value) == expected

    def test_negative_money_keeps_its_sign(self):
        assert research_report._format_money(-1.17e9) == "$-1.17B"
