"""Tests for the thesis conditions and the derived SWOT.

Both modules turn figures into prose, so the risk is prose that does not follow
from the figures. These cases pin the mapping in both directions: a strong
company must not produce weaknesses it does not have, and a loss-making one must
not be described as profitable.
"""

from __future__ import annotations

import pytest

from agents import swot as swot_module
from agents import thesis as thesis_module
from agents.fa_analyst import FundamentalRead, PeerComparison
from agents.insider_analyst import InsiderRead
from agents.ta_analyst import TechnicalRead


def _fundamental(metrics: dict, peers: PeerComparison | None = None) -> FundamentalRead:
    return FundamentalRead(
        symbol="TEST", metrics=metrics, signal="fair", notes="", peer_comparison=peers
    )


def _peers(relative: dict, niche: str = "Test Niche") -> PeerComparison:
    return PeerComparison(
        niche=niche,
        peers_used=["AAA", "BBB"],
        is_sector_fallback=False,
        symbol_metrics={k: 1.0 for k in relative},
        peer_medians={k: 1.0 for k in relative},
        relative=relative,
        verdict="in-line",
        notes="",
    )


def _technical(close: float, sma_200: float) -> TechnicalRead:
    return TechnicalRead(
        symbol="TEST",
        indicators={"close": close, "sma_200": sma_200},
        signal="neutral",
        notes="",
    )


def _insider(buys: int = 0, sells: int = 0, senior_sells: int = 0) -> InsiderRead:
    return InsiderRead(
        symbol="TEST", signal="bearish" if sells else "neutral", lookback_days=180,
        buy_count=buys, sell_count=sells, buy_value=buys * 1000.0,
        sell_value=sells * 1000.0, net_value=(buys - sells) * 1000.0,
        senior_buy_count=0, senior_sell_count=senior_sells,
    )


class TestThesisConditions:
    def test_profitable_company_gets_a_met_earnings_condition(self):
        thesis = thesis_module.build_thesis(
            "TEST", _fundamental({"trailing_eps": 5.0, "profit_margin": 0.25}),
            None, None, None,
        )
        earnings = [c for c in thesis.bull_case if c.metric == "trailing_eps"]
        assert earnings and earnings[0].met is True

    def test_loss_making_company_names_the_turn_as_the_condition(self):
        thesis = thesis_module.build_thesis(
            "TEST", _fundamental({"trailing_eps": -0.31, "forward_eps": 0.90}),
            None, None, None,
        )
        turn = [c for c in thesis.bull_case if c.metric == "trailing_eps"]
        assert turn and turn[0].met is False
        assert "turn" in turn[0].claim.lower()

    def test_premium_valuation_becomes_an_unmet_condition(self):
        thesis = thesis_module.build_thesis(
            "TEST",
            _fundamental({"trailing_eps": 1.0}, _peers({"forward_pe": 5.3})),
            None, None, None,
        )
        valuation = [c for c in thesis.bull_case if c.metric == "forward_pe"]
        assert valuation and valuation[0].met is False

    def test_discount_valuation_becomes_a_met_condition(self):
        thesis = thesis_module.build_thesis(
            "TEST",
            _fundamental({"trailing_eps": 1.0}, _peers({"forward_pe": 0.87})),
            None, None, None,
        )
        valuation = [c for c in thesis.bull_case if c.metric == "forward_pe"]
        assert valuation and valuation[0].met is True

    def test_unknown_conditions_do_not_count_toward_the_tally(self):
        """A condition with no data must not be scored as passed or failed."""
        thesis = thesis_module.build_thesis(
            "TEST", _fundamental({"trailing_eps": 1.0}), None, None, None
        )
        assert thesis.conditions_total == len(
            [c for c in thesis.bull_case if c.met is not None]
        )

    def test_stance_tracks_the_share_of_conditions_met(self):
        strong = thesis_module.build_thesis(
            "TEST",
            _fundamental(
                {"trailing_eps": 5.0, "profit_margin": 0.30, "revenue_growth": 0.40},
                _peers({"forward_pe": 0.8, "revenue_growth": 2.0}),
            ),
            _technical(close=110.0, sma_200=100.0),
            _insider(buys=3),
            None,
        )
        weak = thesis_module.build_thesis(
            "TEST",
            _fundamental(
                {"trailing_eps": -1.0, "profit_margin": -0.20, "revenue_growth": -0.05},
                _peers({"forward_pe": 4.0}),
            ),
            _technical(close=90.0, sma_200=100.0),
            _insider(sells=10, senior_sells=5),
            None,
        )
        assert strong.stance == "constructive"
        assert weak.stance == "skeptical"

    def test_breaks_if_is_populated(self):
        thesis = thesis_module.build_thesis(
            "TEST",
            _fundamental({"trailing_eps": 2.0, "revenue_growth": 0.40, "profit_margin": 0.2}),
            _technical(close=110.0, sma_200=100.0),
            _insider(sells=5, senior_sells=3),
            None,
        )
        assert thesis.breaks_if
        assert any("growth" in b.lower() for b in thesis.breaks_if)

    def test_no_fundamentals_yields_an_empty_thesis(self):
        thesis = thesis_module.build_thesis("TEST", None, None, None, None)
        assert thesis.bull_case == []
        assert "no fundamental data" in thesis.summary.lower()

    def test_every_condition_states_both_claim_and_current(self):
        thesis = thesis_module.build_thesis(
            "TEST",
            _fundamental({"trailing_eps": 1.0, "profit_margin": 0.2, "revenue_growth": 0.2}),
            _technical(close=110.0, sma_200=100.0), _insider(sells=2), None,
        )
        for condition in thesis.bull_case:
            assert condition.claim.strip()
            assert condition.current.strip()


class TestSwot:
    def test_high_margin_is_a_strength(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({"profit_margin": 0.31}), None, None, None
        )
        assert any("margin" in s.metric.lower() for s in swot.strengths)

    def test_negative_margin_is_a_weakness_not_a_strength(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({"profit_margin": -0.15}), None, None, None
        )
        assert any("margin" in w.metric.lower() for w in swot.weaknesses)
        assert not swot.strengths

    def test_loss_making_company_is_flagged(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({"trailing_eps": -0.31}), None, None, None
        )
        assert any("EPS" in w.metric for w in swot.weaknesses)

    def test_peer_premium_is_a_threat(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({}, _peers({"forward_pe": 5.3})), None, None, None
        )
        assert any("P/E" in t.metric for t in swot.threats)

    def test_peer_discount_is_an_opportunity(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({}, _peers({"forward_pe": 0.80})), None, None, None
        )
        assert any("P/E" in o.metric for o in swot.opportunities)

    def test_price_below_long_term_average_is_a_threat(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({}), _technical(close=90.0, sma_200=100.0), None, None
        )
        assert any("200-day" in t.metric for t in swot.threats)

    def test_insider_buying_is_an_opportunity(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({}), None, _insider(buys=4), None
        )
        assert any("insider" in o.metric.lower() for o in swot.opportunities)

    def test_persistent_senior_selling_is_a_threat(self):
        swot = swot_module.build_swot(
            "TEST", _fundamental({}), None, _insider(sells=20, senior_sells=17), None
        )
        assert any("insider" in t.metric.lower() for t in swot.threats)

    def test_every_item_carries_its_metric_and_value(self):
        """A SWOT line without a number behind it is an opinion, not a finding."""
        swot = swot_module.build_swot(
            "TEST",
            _fundamental(
                {"profit_margin": 0.31, "revenue_growth": 0.61, "trailing_eps": -1.0},
                _peers({"forward_pe": 2.0}),
            ),
            _technical(close=90.0, sma_200=100.0),
            _insider(sells=10, senior_sells=5),
            None,
        )
        every = swot.strengths + swot.weaknesses + swot.opportunities + swot.threats
        assert every
        for item in every:
            assert item.metric.strip()
            assert item.value.strip()

    def test_no_fundamentals_yields_an_empty_swot(self):
        assert swot_module.build_swot("TEST", None, None, None, None).is_empty


class TestHtmlRendering:
    def test_report_renders_without_data(self):
        from agents.html_report import render_report
        from agents.research_report import ResearchReport

        html = render_report(
            ResearchReport(symbol="TEST", verdict="inconclusive", score=0.0, confidence=0.0)
        )
        assert "<!doctype html>" in html.lower()
        assert "TEST" in html
        assert "not financial advice" in html

    def test_symbol_is_escaped(self):
        """Report content must not be able to inject markup."""
        from agents.html_report import render_report
        from agents.research_report import ResearchReport

        html = render_report(
            ResearchReport(
                symbol="<script>alert(1)</script>", verdict="mixed",
                score=0.0, confidence=0.5,
            )
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    @pytest.mark.parametrize("score", [-1.0, -0.5, 0.0, 0.43, 1.0])
    def test_gauge_renders_across_the_range(self, score):
        from agents.html_report import _score_gauge_svg

        svg = _score_gauge_svg(score)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_page_carries_the_author_branding(self):
        from agents.html_report import render_report
        from agents.research_report import ResearchReport

        html = render_report(
            ResearchReport(symbol="TEST", verdict="mixed", score=0.0, confidence=0.5)
        )
        assert "NIKOLAY GELSHTEIN" in html
        assert "Nikolay Gelshtein" in html  # title and footer

    def test_no_double_dashes_in_rendered_prose(self):
        """Visible copy uses em dashes; ' -- ' must not reach the page.

        CSS custom properties (--bg) are excluded: there the dashes are syntax.
        """
        from agents.html_report import render_report
        from agents.research_report import ResearchReport

        html = render_report(
            ResearchReport(symbol="TEST", verdict="mixed", score=0.0, confidence=0.5)
        )
        body = html.split("</style>", 1)[-1]
        assert " -- " not in body

    def test_headlines_render_as_links(self):
        from agents.html_report import _sentiment_section
        from agents.news_signal import Headline, NewsSignal, VolumeRead
        from agents.research_report import ResearchReport
        from agents.sentiment_analyst import SentimentRead

        volume = VolumeRead(
            recent_count=0, recent_window_days=3, baseline_count=0,
            baseline_window_days=30, recent_per_day=0.0, baseline_per_day=0.0,
            ratio=1.0, status="unknown", notes="",
        )
        detail = NewsSignal(
            score=0.4, unweighted_score=0.4, article_count=1, dated_count=1,
            effective_sample=1.0, volume=volume,
            top_positive=[
                Headline(
                    title="Stock surges on record profit",
                    url="https://example.com/story",
                    source="Reuters",
                    score=0.8,
                    age_days=0.4,
                )
            ],
        )
        report = ResearchReport(
            symbol="TEST", verdict="positive", score=0.5, confidence=1.0,
            sentiment=SentimentRead(
                symbol="TEST", reddit_score=0.0, news_score=0.4, sample_size=1,
                notes="", signal="bullish", news_sample=1, news_detail=detail,
            ),
        )

        html = _sentiment_section(report)
        assert 'href="https://example.com/story"' in html
        assert 'rel="noopener noreferrer"' in html
        assert "Reuters" in html

    def test_headline_without_url_still_renders(self):
        """A provider may omit the link; the title must survive regardless."""
        from agents.html_report import _sentiment_section
        from agents.news_signal import Headline, NewsSignal, VolumeRead
        from agents.research_report import ResearchReport
        from agents.sentiment_analyst import SentimentRead

        volume = VolumeRead(
            recent_count=0, recent_window_days=3, baseline_count=0,
            baseline_window_days=30, recent_per_day=0.0, baseline_per_day=0.0,
            ratio=1.0, status="unknown", notes="",
        )
        detail = NewsSignal(
            score=-0.4, unweighted_score=-0.4, article_count=1, dated_count=0,
            effective_sample=1.0, volume=volume,
            top_negative=[
                Headline(title="Shares plunge", url="", source="", score=-0.8)
            ],
        )
        report = ResearchReport(
            symbol="TEST", verdict="negative", score=-0.5, confidence=1.0,
            sentiment=SentimentRead(
                symbol="TEST", reddit_score=0.0, news_score=-0.4, sample_size=1,
                notes="", signal="bearish", news_sample=1, news_detail=detail,
            ),
        )

        html = _sentiment_section(report)
        assert "Shares plunge" in html
        assert "<a href" not in html

    def test_headline_markup_is_escaped(self):
        """Titles and URLs come from an external feed: treat both as untrusted."""
        from agents.html_report import _sentiment_section
        from agents.news_signal import Headline, NewsSignal, VolumeRead
        from agents.research_report import ResearchReport
        from agents.sentiment_analyst import SentimentRead

        volume = VolumeRead(
            recent_count=0, recent_window_days=3, baseline_count=0,
            baseline_window_days=30, recent_per_day=0.0, baseline_per_day=0.0,
            ratio=1.0, status="unknown", notes="",
        )
        detail = NewsSignal(
            score=0.4, unweighted_score=0.4, article_count=1, dated_count=1,
            effective_sample=1.0, volume=volume,
            top_positive=[
                Headline(
                    title="<script>alert(1)</script>",
                    url='" onmouseover="alert(1)',
                    source="x",
                    score=0.8,
                )
            ],
        )
        report = ResearchReport(
            symbol="TEST", verdict="positive", score=0.5, confidence=1.0,
            sentiment=SentimentRead(
                symbol="TEST", reddit_score=0.0, news_score=0.4, sample_size=1,
                notes="", signal="bullish", news_sample=1, news_detail=detail,
            ),
        )

        html = _sentiment_section(report)
        assert "<script>" not in html
        assert ' onmouseover="' not in html
