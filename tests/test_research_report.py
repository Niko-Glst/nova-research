"""Tests for the orchestrator's weighting and verdict logic.

Built entirely from hand-made analyst reads, so the combination rules are tested
independently of whether any data provider is reachable.
"""

from __future__ import annotations

import pytest

from datetime import date

from agents import breadth_analyst, flow_analyst, implied_move, insider_analyst, research_report
from agents import sentiment_analyst, ta_analyst
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


def _flows(signal: str) -> flow_analyst.FlowRead:
    return flow_analyst.FlowRead(symbol="TEST", signal=signal, score=float("nan"),
                                 components=())


def _breadth(signal: str) -> breadth_analyst.BreadthRead:
    score = research_report._stance_value(signal)
    return breadth_analyst.BreadthRead(symbol="TEST", signal=signal, score=score,
                                       group=breadth_analyst.BREADTH)


def _market_move() -> implied_move.MarketMove:
    return implied_move.MarketMove(session=date(2026, 10, 23), move=-0.0052, vol_prev=19.03,
                                   implied=0.012, multiple=-0.43, tier=implied_move.NOISE)


@pytest.fixture(autouse=True)
def _no_live_flows(monkeypatch):
    """Flows, breadth and the market move fetch live data; tests fake them (legs unknown)."""
    monkeypatch.setattr(flow_analyst, "analyze", lambda *a, **k: _flows("unknown"))
    monkeypatch.setattr(breadth_analyst, "analyze", lambda *a, **k: _breadth("unknown"))
    monkeypatch.setattr(implied_move, "analyze", lambda *a, **k: _market_move())


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
    def _run(self, monkeypatch, technical, fundamental, sentiment, insider, flows="neutral",
             breadth="neutral"):
        monkeypatch.setattr(flow_analyst, "analyze", lambda *a, **k: _flows(flows))
        monkeypatch.setattr(breadth_analyst, "analyze", lambda *a, **k: _breadth(breadth))
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
        report = self._run(monkeypatch, "bullish", "undervalued", "bullish", "bullish",
                           "bullish", "bullish")
        assert report.verdict == "positive"
        assert report.score == pytest.approx(1.0)
        assert report.confidence == pytest.approx(1.0)

    def test_unanimous_bearish(self, monkeypatch):
        report = self._run(monkeypatch, "bearish", "overvalued", "bearish", "bearish",
                           "bearish", "bearish")
        assert report.verdict == "negative"
        assert report.score == pytest.approx(-1.0)

    def test_known_weighting_case(self, monkeypatch):
        """FA +1.0, TA +0.8, insider -0.6, sentiment, flows and breadth 0 -> +1.2/3.7."""
        report = self._run(monkeypatch, "bullish", "undervalued", "neutral", "bearish")
        assert report.score == pytest.approx(1.2 / 3.7, abs=1e-6)
        assert report.verdict == "positive"

    def test_flows_count_with_weight_0_4(self, monkeypatch):
        """Only flows has a direction: +0.4 / 3.7 = 0.11, not enough for a verdict."""
        assert research_report.WEIGHTS["flows"] == 0.4
        assert sum(research_report.WEIGHTS.values()) == pytest.approx(3.7)
        report = self._run(monkeypatch, "neutral", "fair", "neutral", "neutral", "bullish")
        assert report.score == pytest.approx(0.4 / 3.7, abs=1e-6)
        assert report.verdict == "mixed"
        assert report.flows is not None and report.flows.signal == "bullish"

    def test_breadth_counts_with_weight_0_5(self, monkeypatch):
        """Only breadth has a direction: -0.5 / 3.7 = -0.14, short of a verdict."""
        assert research_report.WEIGHTS["breadth"] == 0.5
        report = self._run(monkeypatch, "neutral", "fair", "neutral", "neutral", "neutral",
                           "bearish")
        assert report.score == pytest.approx(-0.5 / 3.7, abs=1e-6)
        assert report.verdict == "mixed"
        assert report.breadth is not None and report.breadth.signal == "bearish"

    def test_breadth_without_a_call_is_left_out(self, monkeypatch):
        """A breadth leg with no call takes no weight: FA +1 alone stays a full +1.0."""
        report = self._run(monkeypatch, "neutral", "undervalued", "neutral", "neutral",
                           "neutral", "unknown")
        reported = 1.0 + 0.8 + 0.6 + 0.4 + 0.4  # every leg but breadth
        assert report.score == pytest.approx(1.0 / reported, abs=1e-6)
        assert report.confidence == pytest.approx(reported / 3.7, abs=1e-6)

    def test_breadth_gets_the_fundamental_sector_and_cap(self, monkeypatch):
        seen = {}

        def _capture(symbol, sector=None, market_cap=None):
            seen.update(symbol=symbol, sector=sector, market_cap=market_cap)
            return _breadth("neutral")

        monkeypatch.setattr(breadth_analyst, "analyze", _capture)
        monkeypatch.setattr(ta_analyst, "analyze", lambda *a, **k: _technical("neutral"))
        monkeypatch.setattr(research_report.fa_analyst, "analyze", lambda *a, **k: FundamentalRead(
            symbol="TEST", metrics={"market_cap": 7e11}, signal="fair", notes="",
            sector="Financial Services"))
        monkeypatch.setattr(sentiment_analyst, "analyze", lambda *a, **k: _sentiment("neutral"))
        monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: _insider("neutral"))
        research_report.analyze("TEST")
        assert seen == {"symbol": "TEST", "sector": "Financial Services", "market_cap": 7e11}

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
        # Only FA (1.0) and TA (0.8) reported, out of 3.7 total weight.
        assert report.confidence == pytest.approx(1.8 / 3.7, abs=1e-6)
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

        for module in (ta_analyst, sentiment_analyst, insider_analyst, flow_analyst,
                       breadth_analyst, implied_move):
            monkeypatch.setattr(module, "analyze", _boom)
        monkeypatch.setattr(research_report.fa_analyst, "analyze", _boom)

        report = research_report.analyze("TEST")
        assert report.verdict == "inconclusive"
        assert report.confidence == 0.0
        assert len(report.errors) == 7 and "market" in report.errors

    def test_a_ticker_with_no_data_is_inconclusive(self, monkeypatch):
        """ZZZZZ once came out POSITIVE +1.00: only the market-wide flows part had data."""
        def no_data(*args, **kwargs):
            raise ValueError("No price history returned for 'ZZZZZ'")

        market = flow_analyst.FlowComponent("systematic_market", "Market", 1.0, "")
        missing = [flow_analyst.FlowComponent(n, n, None, "")
                   for n in ("issuance", "systematic_stock")]
        parts = (missing[0], market, missing[1])
        signal, score = flow_analyst.classify(parts)
        monkeypatch.setattr(flow_analyst, "analyze", lambda *a, **k: flow_analyst.FlowRead(
            symbol="ZZZZZ", signal=signal, score=score, components=parts))
        monkeypatch.setattr(ta_analyst, "analyze", no_data)
        monkeypatch.setattr(research_report.fa_analyst, "analyze", no_data)
        monkeypatch.setattr(sentiment_analyst, "analyze", lambda *a, **k: _sentiment("unknown"))
        monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: _insider("unknown"))
        report = research_report.analyze("ZZZZZ")
        assert report.verdict == "inconclusive" and report.confidence == 0.0

    def test_market_move_is_shown_not_scored(self, monkeypatch):
        """The market move never changes the score, only adds an evidence line."""
        monkeypatch.setattr(ta_analyst, "analyze", lambda *a, **k: _technical("bullish"))
        monkeypatch.setattr(research_report.fa_analyst, "analyze",
                            lambda *a, **k: _fundamental("undervalued"))
        monkeypatch.setattr(sentiment_analyst, "analyze", lambda *a, **k: _sentiment("neutral"))
        monkeypatch.setattr(insider_analyst, "analyze", lambda *a, **k: _insider("neutral"))
        shown = research_report.analyze("TEST")

        def _offline(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(implied_move, "analyze", _offline)
        hidden = research_report.analyze("TEST")
        assert shown.score == hidden.score and shown.confidence == hidden.confidence
        assert shown.market_move is not None and hidden.market_move is None
        assert any(line.startswith("S&P 500 -0.52%") for line in shown.thesis.evidence)


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
