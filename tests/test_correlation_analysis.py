"""End-to-end tests for the correlation analysis, synchronicity checks and CLI output."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from correlations import synchronicity
from correlations.analysis import load_returns, run_analysis
from correlations.estimators import estimate_covariance, pearson
from correlations.report import format_analysis, to_jsonable

CONFIG = {
    "universes": {
        "demo": {"symbols": ["US1", "US2", "EU1.PA"], "market_proxy": "US1"},
    },
    "exchanges": {
        "": {"currency": "USD", "region": "americas"},
        ".PA": {"currency": "EUR", "region": "europe"},
    },
    "fx_per_eur": {"USD": {"multiply": ["DEXUSEU"]}},
}


def _fake_market(n: int = 900, seed: int = 0):
    """Three correlated assets and an FX series, on slightly different calendars."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n)
    market = rng.normal(0, 0.01, n)
    series = {
        "US1": market + rng.normal(0, 0.005, n),
        "US2": 0.5 * market + rng.normal(0, 0.01, n),
        "EU1.PA": 0.7 * market + rng.normal(0, 0.008, n),
        "FRED:DEXUSEU": rng.normal(0, 0.004, n),
    }
    frames = {}
    for i, (symbol, returns) in enumerate(series.items()):
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=index)
        # Each market skips a different set of days.
        frames[symbol] = prices.drop(prices.index[5 + i :: 97])
    return frames


def _fetcher(frames):
    def fetch(symbol, period):
        return pd.DataFrame({"Close": frames[symbol]})

    return fetch


@pytest.fixture(autouse=True)
def _small_simulation(monkeypatch):
    """The Gaussian benchmarks use 200,000 draws; 20,000 is plenty for wiring tests."""
    from correlations import regimes

    monkeypatch.setattr(regimes, "SIMULATION_DRAWS", 20_000)


@pytest.fixture
def analysis(tmp_path):
    return run_analysis(
        "demo",
        base="EUR",
        winsorize_returns=True,
        fetcher=_fetcher(_fake_market()),
        fx_fetcher=_fetcher(_fake_market()),
        cache_dir=tmp_path,
        config=CONFIG,
        now=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )


class TestRunAnalysis:
    def test_builds_local_and_base_views(self, analysis):
        assert [v.name for v in analysis.views] == ["local", "EUR"]
        base = analysis.view("EUR")
        assert "DEXUSEU" in base.convention
        assert {c.symbol for c in base.conversions} == {"US1", "US2", "EU1.PA"}

    def test_every_frequency_and_method_present(self, analysis):
        for view in analysis.views:
            assert [f.frequency for f in view.frequencies] == ["daily", "weekly"]
            daily = view.frequency("daily")
            assert set(daily.estimates) == {
                "pearson", "spearman", "kendall", "ewma", "ledoit_wolf", "ledoit_wolf_identity",
                "denoised", "pca_factor", "group_factor",
            }
            assert daily.clustering is not None
            assert daily.regimes is not None and view.frequency("weekly").regimes is None
            assert set(daily.winsorized) == {"pearson", "ewma"}
            assert [r.window for r in daily.rolling] == [60, 120, 252]

    def test_no_zero_returns_from_calendar_gaps(self, analysis):
        for view in analysis.views:
            for result in view.frequencies:
                assert sum(result.zero_returns.values()) == 0

    def test_cross_continent_pairs_get_lag_tests(self, analysis):
        lagged = analysis.view("local").frequency("daily").lagged
        assert {(s.asset_a, s.asset_b) for s in lagged} == {("US1", "EU1.PA"), ("US2", "EU1.PA")}

    def test_currency_changes_the_correlation(self, analysis):
        local = analysis.view("local").frequency("daily").estimates["pearson"]
        based = analysis.view("EUR").frequency("daily").estimates["pearson"]
        assert local.pair("US1", "EU1.PA") != pytest.approx(based.pair("US1", "EU1.PA"))

    def test_same_currency_universe_has_one_view(self, tmp_path):
        frames = _fake_market()
        result = run_analysis(
            "demo", base="USD", symbols=["US1", "US2"], fetcher=_fetcher(frames),
            fx_fetcher=_fetcher(frames), cache_dir=tmp_path, config=CONFIG,
        )
        assert [v.name for v in result.views] == ["local"]
        assert any("identical" in note for note in result.notes)

    def test_terminal_report(self, analysis):
        text = format_analysis(analysis)
        for heading in ("LOCAL CURRENCY", "BASE CURRENCY EUR", "Benjamini-Hochberg",
                        "Daily vs weekly", "LIMITATIONS", "CURRENCY EFFECT"):
            assert heading in text

    def test_json_is_valid_and_has_no_nan(self, analysis):
        text = json.dumps(to_jsonable(analysis), allow_nan=False)
        payload = json.loads(text)
        assert payload["base_currency"] == "EUR"
        assert "returns" not in payload["views"][0]["frequencies"][0]
        assert payload["evaluation"]["rankings"][0]["scores"]
        assert "losses" not in payload["evaluation"]

    def test_walk_forward_runs_on_the_investor_view(self, analysis):
        result = analysis.evaluation
        assert result is not None and result.evaluated
        assert "DEXUSEU" in result.currency_convention  # the EUR view, not local
        assert {c.key for c in result.contestants} >= {"sample_252", "ledoit_wolf",
                                                       "group_factor"}
        assert "OUT-OF-SAMPLE" in format_analysis(analysis)

    def test_html_report_and_index(self, analysis, tmp_path):
        from correlations import html as html_report

        page = html_report.render(analysis)
        assert "Which estimator forecasts best (out of sample)" in page
        assert page.count('class="bars"') >= 2  # forecast error and long-only charts
        assert ">nan<" not in page.lower()
        latest = html_report.save(analysis, tmp_path / "reports")
        assert latest.exists()
        assert len(list((tmp_path / "reports" / "archive").glob("*.html"))) == 1
        index = (tmp_path / "reports" / "index.html").read_text(encoding="utf-8")
        assert "Out of sample" in index and "best forecaster" in index

    def test_evaluation_can_be_skipped(self, tmp_path):
        frames = _fake_market()
        result = run_analysis(
            "demo", base="EUR", fetcher=_fetcher(frames), fx_fetcher=_fetcher(frames),
            cache_dir=tmp_path, config=CONFIG, evaluate=False,
        )
        assert result.evaluation is None


class TestLoadReturns:
    def test_same_returns_as_the_analysis(self, analysis, tmp_path):
        frames = _fake_market()
        for base, view in (("EUR", "EUR"), (None, "local")):
            loaded = load_returns(
                ["US1", "US2", "EU1.PA"], base=base, fetcher=_fetcher(frames),
                fx_fetcher=_fetcher(frames), cache_dir=tmp_path / str(base), config=CONFIG,
                now=datetime(2026, 10, 8, tzinfo=timezone.utc),
            )
            expected = analysis.view(view).frequency("daily").returns
            pd.testing.assert_frame_equal(loaded.returns, expected)
            assert loaded.convention == analysis.view(view).convention

    def test_feeds_estimate_covariance(self, tmp_path):
        frames = _fake_market()
        loaded = load_returns(["US1", "EU1.PA"], fetcher=_fetcher(frames),
                              fx_fetcher=_fetcher(frames), cache_dir=tmp_path, config=CONFIG)
        risk = estimate_covariance(loaded.returns)
        assert risk.covariance.shape == (2, 2) and np.isfinite(risk.covariance).all().all()
        assert risk.volatility.between(0.05, 0.5).all()  # annualised, not daily

    def test_needs_two_symbols(self, tmp_path):
        with pytest.raises(ValueError, match="two symbols"):
            load_returns(["US1"], config=CONFIG, cache_dir=tmp_path)


class TestSynchronicity:
    def test_lead_lag_is_detected(self):
        rng = np.random.default_rng(5)
        n = 800
        us = rng.normal(0, 0.01, n)
        eu = np.empty(n)
        eu[0] = 0.0
        eu[1:] = 0.6 * us[:-1] + rng.normal(0, 0.008, n - 1)
        frame = pd.DataFrame({"US": us, "EU": eu}, index=pd.bdate_range("2020-01-01", periods=n))

        stats = synchronicity.lagged_correlations(frame, [("US", "EU")])
        us_leads = next(s for s in stats if s.note == "US leads")
        eu_leads = next(s for s in stats if s.note == "EU leads")
        assert us_leads.correlation > 0.4 and us_leads.significant_adjusted
        # The reverse direction is pure noise. Its significance is a coin with a
        # 5% edge, so assert on size, not on p.
        assert abs(eu_leads.correlation) < 0.15
        assert abs(pearson(frame).pair("US", "EU")) < 0.15  # same-day misses it

    def test_epps_flag(self):
        rng = np.random.default_rng(6)
        index = pd.bdate_range("2020-01-01", periods=300)
        daily = pearson(pd.DataFrame(rng.normal(size=(300, 2)), index=index, columns=["A", "B"]))
        x = rng.normal(size=300)
        weekly = pearson(pd.DataFrame({"A": x, "B": x + 0.3 * rng.normal(size=300)}, index=index))
        flags = synchronicity.epps_flags(daily, weekly, {"A": "americas", "B": "europe"})
        assert len(flags) == 1 and flags[0].cross_continent
        assert flags[0].gap > synchronicity.EPPS_FLAG_THRESHOLD
        assert flags[0].likely_cause == synchronicity.CAUSE_CLOSES

        # Regression: same-region gaps in the EUR view were blamed on close times.
        same = {"A": "americas", "B": "americas"}
        converted = synchronicity.epps_flags(daily, weekly, same, converted=True)
        assert converted[0].likely_cause == synchronicity.CAUSE_FX_TIMING
        local = synchronicity.epps_flags(daily, weekly, same)
        assert local[0].likely_cause == synchronicity.CAUSE_OTHER
