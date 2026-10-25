"""Pair report end to end on synthetic prices, plus its HTML and the CLI switch."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from correlations.pair import analyze_pair, format_pair, summary_lines
from correlations.report import to_jsonable

CONFIG = {
    "universes": {},
    "exchanges": {
        "": {"currency": "USD", "region": "americas"},
        ".PA": {"currency": "EUR", "region": "europe"},
    },
    "symbol_overrides": {"^VIX": {"currency": "USD", "region": "americas"}},
    "fx_per_eur": {"USD": {"multiply": ["DEXUSEU"]}},
}


def _frames(n: int = 2600, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n)
    market = rng.normal(0, 0.01, n)
    returns = {
        "SPY": market,
        "AAA": 0.8 * market + rng.normal(0, 0.008, n),
        "BBB": 0.6 * market + rng.normal(0, 0.012, n),
        "FRED:DEXUSEU": rng.normal(0, 0.004, n),
    }
    frames = {s: pd.Series(100 * np.exp(np.cumsum(r)), index=index) for s, r in returns.items()}
    frames["^VIX"] = pd.Series(20 + 15 * (np.abs(market) > 0.015), index=index, dtype=float)
    return frames


def _fetcher(frames):
    def fetch(symbol, period):
        return pd.DataFrame({"Close": frames[symbol]})

    return fetch


@pytest.fixture(autouse=True)
def _small_simulation(monkeypatch):
    from correlations import regimes

    monkeypatch.setattr(regimes, "SIMULATION_DRAWS", 20_000)


@pytest.fixture
def pair(tmp_path):
    frames = _frames()
    return analyze_pair(
        "aaa", "BBB", base="EUR", fetcher=_fetcher(frames), fx_fetcher=_fetcher(frames),
        cache_dir=tmp_path, config=CONFIG, now=datetime(2026, 10, 16, tzinfo=timezone.utc),
    )


class TestAnalyzePair:
    def test_figures_in_the_base_currency(self, pair):
        assert pair.name == "AAA vs BBB" and pair.view == "EUR"
        full = pair.figure("Full history, daily")
        assert 0.3 < full.value < 0.8 and full.n_observations == pair.n_days
        assert pair.figure("Last 12 months") is not None
        assert any("local" in f.label for f in pair.other_view)

    def test_forecasts_with_a_range(self, pair):
        assert [f.horizon_months for f in pair.forecasts] == [1, 12]
        for f in pair.forecasts:
            assert 0.0 <= f.weight <= 1.0
            assert f.low <= f.forecast <= f.high
            assert f.forecast == pytest.approx(f.weight * f.recent + (1 - f.weight) * f.long_run)

    def test_stress_and_calendar_years(self, pair):
        assert pair.stress is not None and pair.stress.label == "VIX > 30"
        years = [f.label for f in pair.calendar_years]
        assert years[0] == "2018" and len(years) >= 5

    def test_text_json_and_html(self, pair, tmp_path):
        text = format_pair(pair)
        for heading in ("THE NUMBER", "FORECAST", "IN STRESS", "BY CALENDAR YEAR", "LIMITATIONS"):
            assert heading in text
        assert any(t.startswith("Educated guess") for t, _, _ in summary_lines(pair))
        payload = json.loads(json.dumps(to_jsonable(pair), allow_nan=False))
        assert "analysis" not in payload and len(payload["forecasts"]) == 2

        from correlations import html as html_report

        path = html_report.save_pair(pair, tmp_path / "reports")
        assert path.name == "pair_AAA_BBB.html"
        page = path.read_text(encoding="utf-8")
        assert "Forecast, next 12 months" in page and ">nan<" not in page.lower()
        index = (tmp_path / "reports" / "index.html").read_text(encoding="utf-8")
        assert "AAA vs BBB" in index and "12-month forecast" in index

    def test_short_history_gives_no_range_and_says_so(self, tmp_path):
        frames = _frames(n=1500)
        short = analyze_pair("AAA", "BBB", fetcher=_fetcher(frames), fx_fetcher=_fetcher(frames),
                             cache_dir=tmp_path, config=CONFIG)
        twelve = short.forecast(12)
        assert math.isnan(twelve.low) and math.isfinite(twelve.forecast)
        assert any("too few past outcomes" in n for n in short.notes)

    def test_same_symbol_twice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="two different"):
            analyze_pair("AAA", "aaa", config=CONFIG, cache_dir=tmp_path)


class TestCli:
    def test_pair_and_universe_are_alternatives(self):
        from correlation import build_parser

        args = build_parser().parse_args(["--pair", "AAPL", "MSFT"])
        assert args.pair == ["AAPL", "MSFT"] and args.universe is None
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--pair", "AAPL", "MSFT", "--universe", "cross_asset"])
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_figures_are_finite(self, pair):
        assert all(math.isfinite(f.value) for f in pair.figures)
