"""Position-trading rule: weekly swings point in time, conditions, stops, report.

Synthetic prices only; nothing here touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents import position_strategy as ps
from agents import research_report


def _ohlc(close: np.ndarray, start="2022-01-03") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(close))
    close = pd.Series(close, index=days)
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                         "Close": close, "Volume": 1e6})


def _uptrend_with_pullbacks(n=900, seed=1) -> np.ndarray:
    """A rising market with regular dips: legs up, partial retracements, new highs."""
    t = np.arange(n)
    trend = 100 * np.exp(0.0009 * t)
    waves = 1 + 0.08 * np.sin(2 * np.pi * t / 90)
    noise = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.004, n)))
    return trend * waves * noise


class TestWeeklySwings:
    def test_a_swing_is_known_only_after_its_confirming_weeks(self):
        ohlc = _ohlc(_uptrend_with_pullbacks())
        swings = ps.weekly_swings(ohlc)
        assert not swings.empty and set(swings["kind"]) == {"low", "high"}
        lag = (swings["known_on"] - swings["week"]).dt.days
        assert (lag >= 7 * ps.SWING_WEEKS - 4).all()  # two later weeks must have closed

    def test_the_future_never_changes_a_known_swing(self):
        full = _ohlc(_uptrend_with_pullbacks())
        cut = full.iloc[:500]
        early = ps.weekly_swings(cut)
        late = ps.weekly_swings(full)
        known = late[late["known_on"] <= cut.index[-1]]
        pd.testing.assert_frame_equal(early.reset_index(drop=True),
                                      known.reset_index(drop=True))

    def test_leg_and_fibonacci(self):
        leg = ps.Leg(low=100.0, high=200.0, low_week=pd.Timestamp("2024-01-05"),
                     high_week=pd.Timestamp("2024-03-01"))
        assert leg.retracement(150.0) == pytest.approx(0.5)
        assert leg.level(0.618) == pytest.approx(138.2)
        assert leg.extension(1.618) == pytest.approx(261.8)


@pytest.fixture(scope="module")
def result():
    return ps.position_signals(_ohlc(_uptrend_with_pullbacks()), "TEST")


@pytest.fixture(scope="module")
def pb():
    return ps.run("TEST", _ohlc(_uptrend_with_pullbacks()), num_simulations=200)


class TestRule:
    def test_takes_trades_only_when_every_condition_held(self, result):
        entries = result.signals.entries[result.signals.entries]
        assert len(entries) > 0
        frame = result.conditions
        for day in entries.index:
            assert frame.loc[day, "uptrend"] and frame.loc[day, "trigger"]
            window = frame.loc[:day].tail(ps.SETUP_WINDOW)
            assert window["setup"].any()  # trend + Fibonacci zone + support, recently

    def test_bollinger_touch_and_return_is_required(self, result):
        """A close at or below the lower band in the window, and back inside on entry."""
        frame = result.conditions
        for day in result.signals.entries[result.signals.entries].index:
            assert frame.loc[day, "bollinger"]

    def test_never_enters_a_pullback_that_broke_below_61_8(self, result):
        """MU, 28 Oct 2026: momentum turned at 78% retraced; that pullback had failed."""
        frame = result.conditions
        for day in result.signals.entries[result.signals.entries].index:
            assert frame.loc[day, "retracement"] <= ps.FIB_HIGH

    def test_open_trade_is_flagged(self):
        days = pd.bdate_range("2026-01-01", periods=5)
        entries = pd.Series([False, True, False, False, False], index=days)
        exits = pd.Series(False, index=days)
        sig = ps.PositionSignals(ps.StrategySignals("X", entries, exits, ""), pd.DataFrame(),
                                 pd.Series(dtype=float))
        assert sig.open_at_end
        exits.iloc[3] = True
        assert not sig.open_at_end

    def test_entries_and_exits_alternate(self, result):
        e, x = result.signals.entries, result.signals.exits
        events = sorted([(d, "in") for d in e[e].index] + [(d, "out") for d in x[x].index])
        kinds = [k for _, k in events]
        assert all(a != b for a, b in zip(kinds, kinds[1:])) and kinds[0] == "in"

    def test_stop_never_falls_during_a_trade(self, result):
        stops = result.stops
        e, x = result.signals.entries, result.signals.exits
        for start in e[e].index:
            later_exits = x[x].index[x[x].index > start]
            end = later_exits[0] if len(later_exits) else stops.index[-1]
            path = stops.loc[start:end].dropna()
            assert (path.diff().dropna() >= 0).all()

    def test_exit_below_stop_or_200_day(self, result):
        frame = result.conditions
        for day in result.signals.exits[result.signals.exits].index:
            stop = result.stops.loc[day]
            assert frame.loc[day, "close"] < stop or frame.loc[day, "close"] < \
                frame.loc[day, "ma_slow"]

    def test_a_downtrend_never_trades(self):
        falling = _ohlc(100 * np.exp(-0.001 * np.arange(700)) *
                        (1 + 0.05 * np.sin(np.arange(700) / 15)))
        assert not ps.position_signals(falling).signals.entries.any()


class TestReport:
    def test_backtest_and_levels(self, pb):
        assert pb.backtest.num_trades > 0 and pb.monte_carlo is not None
        assert set(pb.levels.moving_averages) == {50, 100, 200}
        lo, mid, up = pb.levels.bollinger
        assert lo < mid < up
        assert set(pb.levels.conditions_now) == set(ps.CONDITIONS)

    def test_sections(self, pb):
        from agents import html_report

        report = research_report.ResearchReport(symbol="TEST", verdict="mixed", score=0.0,
                                                confidence=0.5, position_backtest=pb,
                                                buy_hold_equity=np.linspace(1, 2, 900))
        page = html_report.render_report(report, "2026-10-28")
        assert "Position-trading rule" in page and "Conditions today" in page
        assert "Fibonacci 61.8%" in page and "Bollinger 20/2 (lower / middle / upper)" in page
        text = research_report.format_report(report)
        assert "POSITION-TRADING RULE" in text and "trade(s)" in text
