"""Market breadth leg: RSP/SPY regime, the stock's group, chart and evidence.

All series are synthetic; nothing here touches the network.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agents import breadth_analyst as ba
from agents import charts, flow_evidence, rescore, research_report, signal_snapshots

N = 320  # a year of range plus the turn average's warm-up, with room to spare


def _series(values) -> pd.Series:
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range("2025-06-02", periods=len(values)))


def _zone_turn(bottom: bool) -> pd.Series:
    """A year-long slide into the bottom zone (or climb into the top), then a small turn."""
    start, end = (1.00, 0.90) if bottom else (0.90, 1.00)
    turn = 0.01 if bottom else -0.01
    return _series(np.concatenate([np.linspace(start, end, N - 50), np.full(40, end),
                                   np.full(10, end + turn)]))


def _broadening() -> pd.Series:
    """Up to the 1-year high, down to the low, then a zigzagging climb back to mid-range."""
    zigzag = 0.004 * (-1.0) ** np.arange(100)
    return _series(np.concatenate([np.linspace(0.90, 1.00, 120), np.linspace(1.00, 0.90, 100),
                                   np.linspace(0.90, 0.95, 100) + zigzag]))


def _frames(ratio: pd.Series) -> dict[str, pd.DataFrame]:
    index = ratio.index.tz_localize("America/New_York")  # yfinance hands back tz-aware dates
    return {ba.EQUAL_WEIGHT: pd.DataFrame({"Close": 150.0 * ratio.to_numpy()}, index=index),
            ba.CAP_WEIGHT: pd.DataFrame({"Close": np.full(len(ratio), 150.0)}, index=index)}


@pytest.fixture
def fake_prices(monkeypatch):
    def install(ratio: pd.Series) -> None:
        frames = _frames(ratio)
        monkeypatch.setattr(ba, "fetch_price_history", lambda symbol, period=None: frames[symbol])
    return install


class TestConfig:
    def test_rule_numbers_come_from_config(self):
        assert (ba.EQUAL_WEIGHT, ba.CAP_WEIGHT) == ("RSP", "SPY")
        assert (ba.RANGE_DAYS, ba.ZONE, ba.TURN_MA, ba.TREND_DAYS) == (252, 0.20, 50, 63)
        assert ba.CHART_VIEWS == {"1Y": 252, "3M": 63, "1M": 21}
        assert ba.TREND_LABEL == "3-month trend line"


class TestTrendLine:
    def test_slope_of_a_steady_climb(self):
        values = _series(0.28 * np.exp(0.001 * np.arange(100)))
        slope, fitted = ba.trend_line(values)
        assert slope == pytest.approx(0.001, rel=1e-6)
        assert len(fitted) == ba.TREND_DAYS
        assert fitted[-1] == pytest.approx(values.iloc[-1], rel=1e-9)

    def test_zigzags_do_not_hide_the_direction(self):
        """Up most of three months with a short pullback at the end: still rising."""
        climb = np.linspace(0.90, 0.96, 92) + 0.004 * (-1.0) ** np.arange(92)
        values = _series(np.concatenate([climb, np.linspace(0.96, 0.95, 8)]))
        slope, _ = ba.trend_line(values)
        assert slope > 0

    def test_trend_change_is_the_line_over_its_window(self):
        regime = ba.regime_from_ratio(_broadening())
        assert regime.trend_change == pytest.approx(math.expm1(regime.trend_slope * 62))


class TestRegime:
    def test_bottom_zone_with_turn_favours_breadth(self):
        regime = ba.regime_from_ratio(_zone_turn(bottom=True))
        assert regime.state == "bottom_turn" and regime.regime == 1.0
        assert regime.position <= ba.ZONE and regime.ratio > regime.ma_turn

    def test_bottom_zone_without_turn_waits(self):
        regime = ba.regime_from_ratio(_series(np.linspace(1.00, 0.90, N)))
        assert regime.state == "bottom_waiting" and regime.regime == 0.0
        assert regime.position == pytest.approx(0.0)

    def test_top_zone_with_turn_favours_the_leaders(self):
        regime = ba.regime_from_ratio(_zone_turn(bottom=False))
        assert regime.state == "top_turn" and regime.regime == -1.0
        assert regime.position >= 1 - ba.ZONE and regime.ratio < regime.ma_turn

    def test_top_zone_without_turn_waits_with_a_plain_zero(self):
        regime = ba.regime_from_ratio(_series(np.linspace(0.90, 1.00, N)))
        assert regime.state == "top_waiting" and regime.regime == 0.0
        assert math.copysign(1.0, regime.regime) == 1.0  # never -0.0

    def test_between_zones_a_rising_trend_is_broadening(self):
        regime = ba.regime_from_ratio(_broadening())
        assert ba.ZONE < regime.position < 1 - ba.ZONE
        assert regime.state == "trend_broadening" and regime.regime == 1.0

    def test_between_zones_a_falling_trend_is_narrowing(self):
        regime = ba.regime_from_ratio(1.0 / _broadening())
        assert ba.ZONE < regime.position < 1 - ba.ZONE
        assert regime.state == "trend_narrowing" and regime.regime == -1.0

    def test_only_history_up_to_the_last_date_is_used(self):
        full = _broadening()
        early = ba.regime_from_ratio(full.iloc[:260])
        assert early.ratio == full.iloc[259]

    def test_too_short_or_flat_history_gives_no_regime(self):
        assert ba.regime_from_ratio(_series(np.linspace(0.9, 1.0, 100))) is None
        assert ba.regime_from_ratio(_series(np.full(N, 0.28))) is None

    def test_describe_states_the_numbers(self):
        text = ba.regime_from_ratio(_broadening()).describe()
        assert text.startswith("RSP/SPY ") and "252-day range" in text
        assert "63-day trend line is rising" in text and "50-day average" in text


class TestStockGroup:
    @pytest.mark.parametrize(("sector", "cap", "group"), [
        ("Healthcare", 5e9, ba.BREADTH),
        ("Financial Services", 7e11, ba.BREADTH),
        ("Utilities", None, ba.BREADTH),
        ("Technology", 3e12, ba.LEADER),
        ("Communication Services", 5e11, ba.LEADER),
        ("Consumer Cyclical", 2e12, ba.LEADER),
        ("Technology", 4.99e11, ba.NEUTRAL),
        ("Technology", None, ba.NEUTRAL),
        ("Energy", 5e11, ba.NEUTRAL),
        ("", None, ba.NEUTRAL),
    ])
    def test_sector_and_size(self, sector, cap, group):
        assert ba.classify_stock(sector, cap) == group

    def test_leaders_score_the_opposite(self):
        assert ba.score_for(ba.BREADTH, 1.0) == 1.0
        assert ba.score_for(ba.LEADER, 1.0) == -1.0
        assert ba.score_for(ba.LEADER, -1.0) == 1.0
        assert ba.score_for(ba.NEUTRAL, 1.0) is None

    @pytest.mark.parametrize("group", [ba.BREADTH, ba.LEADER, ba.NEUTRAL])
    def test_no_call_gives_no_score(self, group):
        """Owner 2026-10-25: waiting in a zone leaves the leg out, it is not neutral."""
        assert ba.score_for(group, 0.0) is None
        assert ba.stance(ba.score_for(group, 0.0)) == "unknown"

    def test_stance(self):
        assert [ba.stance(s) for s in (1.0, 0.0, -1.0, None, float("nan"))] == \
            ["bullish", "neutral", "bearish", "unknown", "unknown"]


class TestAnalyze:
    def test_breadth_stock_in_a_broadening_market(self, fake_prices):
        fake_prices(_broadening())
        read = ba.analyze("jpm", sector="Financial Services", market_cap=7e11)
        assert (read.symbol, read.group, read.signal, read.score) == \
            ("JPM", ba.BREADTH, "bullish", 1.0)
        assert read.regime.state == "trend_broadening"
        assert len(read.history) == max(ba.CHART_VIEWS.values()) + ba.TURN_MA
        assert read.history.index.tz is None
        assert read.reasons[0] == read.regime.describe()

    def test_leader_in_a_broadening_market_reads_bearish(self, fake_prices):
        fake_prices(_broadening())
        read = ba.analyze("MSFT", sector="Technology", market_cap=3e12)
        assert (read.group, read.signal, read.score) == (ba.LEADER, "bearish", -1.0)
        assert "opposite of the regime" in read.reasons[1]

    def test_other_stocks_get_no_score(self, fake_prices):
        fake_prices(_broadening())
        read = ba.analyze("XOM", sector="Energy", market_cap=4e11)
        assert (read.group, read.signal, read.score) == (ba.NEUTRAL, "unknown", None)
        assert read.regime is not None  # the market reading is still shown
        assert "left out of this verdict" in read.reasons[1]

    def test_waiting_in_a_zone_leaves_the_leg_out(self, fake_prices):
        fake_prices(_series(np.linspace(1.00, 0.90, N)))  # bottom zone, no turn yet
        read = ba.analyze("JPM", sector="Financial Services", market_cap=7e11)
        assert read.regime.state == "bottom_waiting"
        assert (read.signal, read.score) == ("unknown", None)
        assert "left out of this verdict" in read.reasons[-1]

    def test_short_history_is_unknown(self, fake_prices):
        fake_prices(_series(np.linspace(0.9, 1.0, 100)))
        read = ba.analyze("JPM", sector="Financial Services", market_cap=7e11)
        assert read.signal == "unknown" and read.regime is None
        assert read.reasons[0].startswith("Not enough RSP/SPY history")


class TestPresentation:
    @pytest.fixture
    def read(self, fake_prices):
        fake_prices(_broadening())
        return ba.analyze("JPM", sector="Financial Services", market_cap=7e11)

    def test_chart_has_three_zoom_views_and_no_script(self, read):
        r = read.regime
        _, fitted = ba.trend_line(read.history)
        trend = dict(zip(read.history.index[-len(fitted):], fitted))
        page = charts.breadth_ratio_chart(read.history, r.low, r.high, ba.ZONE, ba.TURN_MA,
                                          trend, ba.CHART_VIEWS, ba.TREND_LABEL)
        assert page.count("<svg") == 3 and "<script" not in page
        assert page.count(" checked") == 1 and 'id="bz-1Y" class="bz-radio" checked' in page
        assert "3-month trend line" in page and "50-day average" in page
        assert "Top / bottom 20% of the 1-year range" in page
        views = page.split('<div class="bz-view ')[1:]
        assert [v.count('class="bz-hover"') for v in views] == [252, 63, 21]
        assert all('class="bz-trend"' in v and 'class="bz-ma"' in v for v in views)

    def test_html_report_shows_the_section_and_chart(self, read):
        from agents import html_report

        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.1,
                                                confidence=0.5, breadth=read,
                                                model_version="2.0.0")
        page = html_report.render_report(report, "2026-10-25 12:00")
        assert "Market breadth" in page and "bz-chart" in page
        assert "3-month trend line" in page and "trend broadening" in page

    def test_terminal_report_has_a_breadth_section(self, read):
        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.1,
                                                confidence=0.5, breadth=read)
        text = research_report.format_report(report)
        assert "MARKET BREADTH -> BULLISH" in text and "Group: breadth" in text

    def test_evidence_line_has_the_numbers_and_group(self, read):
        line = flow_evidence.breadth_line(read)
        r = read.regime
        assert line.startswith(f"RSP/SPY {r.ratio:.4f} at {r.position:.0%} of its 252-day range")
        assert f"63-day trend line {r.trend_change:+.1%}" in line
        assert "regime +1 (trend broadening)" in line and "breadth stock, leg score +1" in line

    def test_evidence_line_without_a_regime(self):
        read = ba.BreadthRead(symbol="JPM", signal="unknown", score=None, group=ba.BREADTH)
        assert "not enough history" in flow_evidence.breadth_line(read)

    def test_snapshot_keeps_the_inputs_and_rescore_reproduces_the_stance(self, read):
        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.1,
                                                confidence=0.5, breadth=read,
                                                model_version="2.0.0")
        row = signal_snapshots.snapshot_row(report, datetime(2026, 10, 25, tzinfo=timezone.utc))
        assert (row["breadth_signal"], row["breadth_group"]) == ("bullish", ba.BREADTH)
        assert float(row["breadth_ratio"]) == read.regime.ratio
        assert float(row["breadth_trend_slope"]) == read.regime.trend_slope
        assert rescore.breadth_stance(row) == read.signal


def test_a_small_market_cap_is_not_rounded_to_a_billion():
    """XFAB.PA, 2026-11-02: a 686M market cap read "$1B"."""
    assert "market cap $686M" in ba._group_text(ba.NEUTRAL, "Technology", 685_817_600)
    assert "market cap $12B" in ba._group_text(ba.NEUTRAL, "Technology", 12.1e9)
