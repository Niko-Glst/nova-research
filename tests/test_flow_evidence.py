"""Flow evidence lines in the thesis and SWOT: numbers, sources and flags, never verdicts."""

from __future__ import annotations

from types import MappingProxyType

from agents import flow_analyst as fl
from agents import flow_evidence as fe
from agents import research_report, swot as swot_module, thesis as thesis_module
from agents.fa_analyst import FundamentalRead
from agents.ta_analyst import TechnicalRead

REFERENCE = {f"T{i}": i / 100 - 0.05 for i in range(20)}  # -5% .. +14%


def _flows(ni=0.171, flag=fl.CONFIRMED_ZERO, source="SEC 10-K filed 2026-02-23"):
    issuance = fl.FlowComponent(
        "issuance", "Net issuance", fl.issuance_score(ni),
        f"Net issuance {ni:+.1%} over the year to 2025-12-31 ({source}): shares grew.",
        MappingProxyType({"net_issuance": ni}), flag)
    stock = fl.FlowComponent("systematic_stock", "Stock", 0.4, "",
                             MappingProxyType({"vol_ratio": 0.79, "z": -1.6}),
                             "vol_estimator: yang_zhang")
    market = fl.FlowComponent("systematic_market", "Market", 1.0, "",
                              MappingProxyType({"return_1m": 0.0002, "return_3m": 0.051,
                                                "return_12m": 0.165, "vol_ratio": 0.72}))
    return fl.FlowRead(symbol="IBRX", signal="neutral", score=0.1,
                       components=(issuance, market, stock))


def test_issuance_line_has_number_rank_source_and_flag():
    lines = fe.evidence_lines(_flows(), REFERENCE)
    assert lines[0] == ("Net issuance +17.1% YoY (top decile among 20 us_large_caps stocks, "
                        "source: share count from SEC 10-K filed 2026-02-23; "
                        "buyback line confirmed_zero)")
    assert any("yang_zhang" in line and "0.79" in line for line in lines)
    assert any(line.startswith("SPY trend") for line in lines)


def test_without_backfill_the_rank_is_not_guessed():
    assert "rank unavailable" in fe.evidence_lines(_flows(), None)[0]


def test_yfinance_source_says_unconfirmed_and_conflict_says_left_out():
    line = fe.evidence_lines(_flows(-0.05, fl.DATA_CONFLICT, "yfinance annual balance sheet"),
                             REFERENCE)[0]
    assert "not confirmed by SEC" in line and "left out of the score" in line
    assert "bottom decile" in line


def test_evidence_never_changes_the_thesis_or_swot():
    fundamental = FundamentalRead(symbol="IBRX", metrics={"profit_margin": -5.9,
                                                          "revenue_growth": 0.94},
                                  signal="overvalued", notes="")
    technical = TechnicalRead(symbol="IBRX", indicators={"close": 8.6, "sma_200": 7.7},
                              signal="neutral", notes="")
    lines = fe.evidence_lines(_flows(), REFERENCE)
    plain = thesis_module.build_thesis("IBRX", fundamental, technical, None, None)
    fed = thesis_module.build_thesis("IBRX", fundamental, technical, None, None, lines)
    assert (fed.summary, fed.stance, fed.bull_case, fed.breaks_if) == \
        (plain.summary, plain.stance, plain.bull_case, plain.breaks_if)
    assert fed.evidence == lines and plain.evidence == []
    swot_plain = swot_module.build_swot("IBRX", fundamental, technical, None, None)
    swot_fed = swot_module.build_swot("IBRX", fundamental, technical, None, None, lines)
    assert (swot_fed.strengths, swot_fed.weaknesses, swot_fed.opportunities,
            swot_fed.threats) == (swot_plain.strengths, swot_plain.weaknesses,
                                  swot_plain.opportunities, swot_plain.threats)
    assert swot_fed.evidence == lines


def test_reports_show_the_evidence(monkeypatch):
    from agents import html_report

    fundamental = FundamentalRead(symbol="IBRX", metrics={"profit_margin": -5.9},
                                  signal="overvalued", notes="")
    lines = fe.evidence_lines(_flows(), REFERENCE)
    report = research_report.ResearchReport(
        symbol="IBRX", verdict="negative", score=-0.37, confidence=1.0,
        thesis=thesis_module.build_thesis("IBRX", fundamental, None, None, None, lines),
        swot=swot_module.build_swot("IBRX", fundamental, None, None, None, lines),
        model_version="2.0.0")
    text = research_report.format_report(report)
    assert "FLOW EVIDENCE (inputs, not verdicts)" in text and "+17.1% YoY" in text
    page = html_report.render_report(report, "2026-10-25 12:00")
    assert "Flow evidence" in page and "top decile" in page
