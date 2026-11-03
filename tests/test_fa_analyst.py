"""Tests for fundamental analysis: derived ratios, peer resolution, verdicts."""

from __future__ import annotations

import pytest

from agents import fa_analyst as fa


@pytest.fixture(autouse=True)
def custom_peers_file(tmp_path, monkeypatch):
    """Keep the tests away from the real choices in config/custom_peers.yaml."""
    path = tmp_path / "custom_peers.yaml"
    monkeypatch.setattr(fa, "CUSTOM_PEERS_PATH", path)
    return path


class TestCustomPeers:
    METRICS = {"_sector": "Healthcare", "_industry": "No Such Industry", "price_to_sales": 2.0}

    @pytest.fixture
    def fetched(self, monkeypatch):
        """Offline peers, and the tickers that were looked up."""
        calls = []

        def fetch(symbol):
            calls.append(symbol)
            return {"price_to_sales": 4.0}

        monkeypatch.setattr(fa, "fetch_fundamentals", fetch)
        return calls

    def test_no_file_means_no_choices(self):
        assert fa.load_custom_peers() == {}

    def test_saving_keeps_the_other_companies_and_the_header(self, custom_peers_file):
        fa.save_custom_peers("aaa", ["BBB", "NO"])  # NO must not come back as False
        fa.save_custom_peers("CCC", ["DDD"])
        assert fa.load_custom_peers() == {"AAA": ["BBB", "NO"], "CCC": ["DDD"]}
        assert custom_peers_file.read_text(encoding="utf-8").startswith("# Peers chosen")

    def test_the_shipped_file_loads_empty(self, monkeypatch):
        monkeypatch.setattr(fa, "CUSTOM_PEERS_PATH", fa.ROOT_DIR / "config" / "custom_peers.yaml")
        assert isinstance(fa.load_custom_peers(), dict)

    def test_a_malformed_file_says_how_it_should_look(self, custom_peers_file):
        custom_peers_file.write_text("AAA: BBB\n", encoding="utf-8")
        with pytest.raises(ValueError, match="SYMBOL"):
            fa.load_custom_peers()

    def test_chosen_peers_go_before_the_sector_and_are_not_asked_again(self, fetched):
        fa.save_custom_peers("XYZ", ["XYZ", "PEER"])
        asked = []
        result = fa.compare_to_peers("XYZ", self.METRICS, lambda *a: asked.append(a))
        assert not asked and fetched == ["PEER"]
        assert result.is_custom and not result.is_sector_fallback
        assert result.niche == "No Such Industry" and "chosen" in result.notes

    def test_chosen_peers_go_before_a_declared_group(self, fetched):
        fa.save_custom_peers("XYZ", ["PEER"])
        fa.compare_to_peers("XYZ", {"_sector": "Technology", "_industry": "Semiconductors"})
        assert fetched == ["PEER"]

    def test_a_company_on_the_sector_list_is_asked_and_the_answer_is_used(self, fetched):
        asked = []

        def ask(symbol, niche, suggested):
            asked.append((symbol, niche, bool(suggested)))
            return ["PEER"]

        result = fa.compare_to_peers("XYZ", self.METRICS, ask)
        assert asked == [("XYZ", "Healthcare", True)]
        assert fetched == ["PEER"] and result.is_custom and result.verdict == "cheap"

    def test_no_answer_keeps_the_sector(self, fetched):
        result = fa.compare_to_peers("XYZ", self.METRICS, lambda *a: None)
        assert result.is_sector_fallback and not result.is_custom and result.peers_used

    def test_a_company_without_any_peers_is_asked_too(self, fetched):
        metrics = {"_sector": "Nonexistent", "_industry": "Nonexistent"}
        asked = []
        result = fa.compare_to_peers("XYZ", metrics, lambda *a: asked.append(a))
        assert asked == [("XYZ", "Nonexistent", [])]
        assert result.verdict == "unknown" and "custom_peers.yaml" in result.notes

    def test_a_company_with_its_own_group_is_not_asked(self, fetched):
        asked = []
        fa.compare_to_peers("XYZ", {"_sector": "Technology", "_industry": "Semiconductors"},
                            lambda *a: asked.append(a))
        assert not asked


class TestNormalizeKey:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Software - Infrastructure", "software_infrastructure"),
            ("Consumer Electronics", "consumer_electronics"),
            ("Internet Content & Information", "internet_content_information"),
            ("Oil & Gas Integrated", "oil_gas_integrated"),
            ("", ""),
        ],
    )
    def test_labels_become_config_keys(self, label, expected):
        assert fa._normalize_key(label) == expected


class TestComputeRatios:
    def test_implied_eps_growth(self):
        derived = fa.compute_ratios({"trailing_eps": 10.0, "forward_eps": 12.0})
        assert derived["eps_growth_implied"] == pytest.approx(20.0)

    def test_earnings_yield_inverts_pe(self):
        derived = fa.compute_ratios({"trailing_pe": 25.0})
        assert derived["earnings_yield_pct"] == pytest.approx(4.0)

    def test_fcf_yield(self):
        derived = fa.compute_ratios({"market_cap": 1_000.0, "free_cashflow": 50.0})
        assert derived["fcf_yield_pct"] == pytest.approx(5.0)

    def test_negative_eps_yields_no_growth_figure(self):
        """Growth off a negative base is meaningless, so it is omitted."""
        derived = fa.compute_ratios({"trailing_eps": -2.0, "forward_eps": 1.0})
        assert "eps_growth_implied" not in derived

    def test_missing_inputs_produce_no_derived_values(self):
        assert fa.compute_ratios({}) == {}


class TestPeerComparability:
    def test_negative_multiples_are_excluded(self):
        """A loss-making peer's P/E would drag the median toward nonsense."""
        assert fa._is_comparable("trailing_pe", -15.0) is False
        assert fa._is_comparable("trailing_pe", 15.0) is True

    def test_negative_quality_metrics_are_kept(self):
        """A negative margin is real information and belongs in the median."""
        assert fa._is_comparable("profit_margin", -0.20) is True

    def test_missing_values_are_excluded(self):
        assert fa._is_comparable("profit_margin", None) is False


class TestNegativeMultiplesInComparison:
    """A loss-making company must not read as 'cheap' on a negative multiple.

    Regression: a forward P/E of -8.99 against a peer median of 9.70 produced
    -0.93x, which the verdict logic counted as trading below the median.
    """

    def test_negative_own_multiple_is_excluded(self, monkeypatch):
        def _fake_fetch(symbol):
            return {
                "trailing_pe": 10.0,
                "forward_pe": 9.7,
                "price_to_book": 8.4,
                "profit_margin": 0.01,
            }

        monkeypatch.setattr(fa, "fetch_fundamentals", _fake_fetch)

        comparison = fa.compare_to_peers(
            "RIVN",
            {
                "_sector": "Consumer Cyclical",
                "_industry": "Auto Manufacturers",
                "forward_pe": -8.99,  # loss-making
                "price_to_book": 4.28,
                "profit_margin": -0.55,
            },
        )

        assert "forward_pe" not in comparison.relative
        assert "price_to_book" in comparison.relative
        assert "forward_pe" in comparison.notes

    def test_negative_quality_metrics_still_compare(self, monkeypatch):
        """A negative margin is real information, unlike a negative P/E."""
        monkeypatch.setattr(
            fa, "fetch_fundamentals", lambda symbol: {"profit_margin": 0.01}
        )
        comparison = fa.compare_to_peers(
            "RIVN",
            {
                "_sector": "Consumer Cyclical",
                "_industry": "Auto Manufacturers",
                "profit_margin": -0.55,
            },
        )
        assert "profit_margin" in comparison.relative


class TestResolvePeers:
    def test_niche_group_is_preferred(self):
        peers, niche, fallback = fa._resolve_peers("AAPL", "Technology", "Consumer Electronics")
        assert fallback is False
        assert niche == "Consumer Electronics"
        assert "SONY" in peers

    def test_symbol_is_never_its_own_peer(self):
        peers, _, _ = fa._resolve_peers("AAPL", "Technology", "Consumer Electronics")
        assert "AAPL" not in peers

    def test_unknown_industry_falls_back_to_sector(self):
        peers, niche, fallback = fa._resolve_peers("XYZ", "Technology", "Nonexistent Industry")
        assert fallback is True
        assert niche == "Technology"
        assert peers

    def test_unknown_sector_yields_no_peers(self):
        peers, _, fallback = fa._resolve_peers("XYZ", "Nonexistent", "Nonexistent")
        assert peers == []
        assert fallback is True

    def test_peer_count_respects_the_configured_cap(self):
        from config.settings import load_allowed_sources

        cap = int(load_allowed_sources().get("max_peers", 6))
        peers, _, _ = fa._resolve_peers("ZZZZ", "Technology", "Semiconductors")
        assert len(peers) <= cap


def _peer_comparison(verdict: str) -> fa.PeerComparison:
    return fa.PeerComparison(
        niche="Test Niche",
        peers_used=["AAA", "BBB"],
        is_sector_fallback=False,
        symbol_metrics={},
        peer_medians={},
        relative={},
        verdict=verdict,
        notes="",
    )


class TestClassify:
    def test_strong_fundamentals_read_undervalued(self):
        metrics = {
            "trailing_eps": 5.0,
            "eps_growth_implied": 20.0,
            "profit_margin": 0.25,
            "revenue_growth": 0.20,
        }
        signal, reasons = fa._classify(metrics, _peer_comparison("cheap"))
        assert signal == "undervalued"
        assert reasons

    def test_weak_fundamentals_read_overvalued(self):
        metrics = {
            "trailing_eps": -1.0,
            "eps_growth_implied": -20.0,
            "profit_margin": -0.10,
            "revenue_growth": -0.05,
        }
        signal, _ = fa._classify(metrics, _peer_comparison("expensive"))
        assert signal == "overvalued"

    def test_peer_verdict_outweighs_a_single_metric(self):
        """Relative valuation carries double weight, by design."""
        metrics = {"trailing_eps": 5.0, "profit_margin": 0.25}
        cheap, _ = fa._classify(metrics, _peer_comparison("cheap"))
        expensive, _ = fa._classify(metrics, _peer_comparison("expensive"))
        assert cheap == "undervalued"
        assert expensive != "undervalued"

    def test_no_data_reads_fair(self):
        signal, _ = fa._classify({}, None)
        assert signal == "fair"

    def test_high_leverage_counts_against(self):
        _, reasons = fa._classify({"debt_to_equity": 350.0}, None)
        assert any("leverage" in r.lower() for r in reasons)


class TestPlaceholderMargins:
    """Regression: yfinance reports a margin it cannot compute as exactly 0.0.

    Values are ImmunityBio (IBRX) and Summit Therapeutics (SMMT) as yfinance
    returned them on 2026-10-17.
    """

    IBRX = {
        "symbol": "IBRX", "shortName": "ImmunityBio, Inc.", "industry": "Biotechnology",
        "sector": "Healthcare", "totalRevenue": 165_800_000, "profitMargins": 0.0,
        "netIncomeToCommon": -992_356_992, "operatingMargins": -1.20408,
    }

    def test_zero_net_margin_is_recomputed_from_net_income(self):
        out = fa.extract_fundamentals(self.IBRX, "IBRX")
        assert out["profit_margin"] == pytest.approx(-992_356_992 / 165_800_000)
        assert out["operating_margin"] == pytest.approx(-1.20408)  # a real value: untouched
        assert "recomputed" in out["_data_notes"]

    def test_the_repaired_margin_now_counts_against(self):
        out = fa.extract_fundamentals(self.IBRX, "IBRX")
        _, reasons = fa._classify({k: v for k, v in out.items() if not k.startswith("_")}, None)
        assert any("Negative net margin" in r for r in reasons)

    def test_margins_without_revenue_are_dropped(self):
        smmt = {"symbol": "SMMT", "shortName": "Summit", "totalRevenue": 0,
                "profitMargins": 0.0, "operatingMargins": 0.0, "netIncomeToCommon": -856e6}
        out = fa.extract_fundamentals(smmt, "SMMT")
        assert "profit_margin" not in out and "operating_margin" not in out
        assert "undefined" in out["_data_notes"]

    def test_zero_margin_without_net_income_is_missing(self):
        info = {**self.IBRX}
        del info["netIncomeToCommon"]
        assert "profit_margin" not in fa.extract_fundamentals(info, "IBRX")

    def test_real_margins_are_untouched(self):
        info = {"symbol": "AAPL", "totalRevenue": 4e11, "profitMargins": 0.24,
                "grossMargins": 0.46, "operatingMargins": 0.31, "netIncomeToCommon": 9.6e10}
        out = fa.extract_fundamentals(info, "AAPL")
        assert (out["profit_margin"], out["gross_margin"], out["operating_margin"]) == (
            0.24, 0.46, 0.31)
        assert out["_data_notes"] == ""
