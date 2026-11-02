"""Track record: recorded verdicts settled against the stock's local index.

Hand-made prices only; nothing here touches the network.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import research_report, track_record as tr

DAYS = pd.bdate_range("2026-09-01", periods=200)


def _prices(start: float, daily: float) -> pd.Series:
    return pd.Series(start * (1 + daily) ** np.arange(len(DAYS)), index=DAYS)


PRICES = {"BESI.AS": _prices(100, 0.002), "^AEX": _prices(900, 0.001),
          "META": _prices(600, -0.001), "SPY": _prices(500, 0.0005)}


def _row(symbol, recorded_at, verdict="positive", score="0.4", model="3.0.0"):
    return {"symbol": symbol, "recorded_at": recorded_at, "verdict": verdict, "score": score,
            "model_version": model}


def _loader(symbol):
    return PRICES[symbol]


class TestSettleRow:
    def test_entry_is_the_first_close_after_the_run_day(self):
        call = tr.settle_row(_row("BESI.AS", "2026-09-04T19:30:00+00:00"),
                             PRICES["BESI.AS"], PRICES["^AEX"], "^AEX")
        assert call.entry_date == date(2026, 9, 7)  # Friday's run -> Monday's close
        assert call.entry_price == pytest.approx(PRICES["BESI.AS"].loc["2026-09-07"])

    def test_returns_and_excess_over_each_horizon(self):
        call = tr.settle_row(_row("BESI.AS", "2026-09-04T19:30:00+00:00"),
                             PRICES["BESI.AS"], PRICES["^AEX"], "^AEX")
        week = call.outcomes[0]
        assert week.label == "1W" and week.sessions == 5
        assert week.stock_return == pytest.approx(1.002 ** 5 - 1)
        assert week.benchmark_return == pytest.approx(1.001 ** 5 - 1)
        assert call.hit(week) is True  # positive call, stock beat the AEX

    def test_horizons_not_reached_are_pending(self):
        run_day = DAYS[-12].date().isoformat()  # 11 sessions left: past 1W, short of 1M
        call = tr.settle_row(_row("BESI.AS", f"{run_day}T10:00:00+00:00"),
                             PRICES["BESI.AS"], PRICES["^AEX"], "^AEX")
        assert [o.pending for o in call.outcomes] == [False, True, True, True]
        assert call.hit(call.outcomes[1]) is None

    def test_a_run_after_the_last_close_has_no_entry_yet(self):
        call = tr.settle_row(_row("BESI.AS", "2030-01-01T10:00:00+00:00"),
                             PRICES["BESI.AS"], PRICES["^AEX"], "^AEX")
        assert call.entry_date is None and "No close since the run" in call.note
        assert all(o.pending for o in call.outcomes)

    def test_negative_call_is_right_when_the_stock_trails(self):
        call = tr.settle_row(_row("META", "2026-09-04T19:30:00+00:00", "negative", "-0.3"),
                             PRICES["META"], PRICES["SPY"], "SPY")
        assert call.hit(call.outcomes[0]) is True

    def test_mixed_takes_no_side(self):
        call = tr.settle_row(_row("META", "2026-09-04T19:30:00+00:00", "mixed", "0.08"),
                             PRICES["META"], PRICES["SPY"], "SPY")
        assert call.hit(call.outcomes[0]) is None

    def test_no_benchmark_still_shows_the_stock_return(self):
        call = tr.settle_row(_row("EQNR.OL", "2026-09-04T19:30:00+00:00"),
                             PRICES["META"], None, None)
        week = call.outcomes[0]
        assert math.isnan(week.excess) and call.hit(week) is None
        assert "no benchmark" in tr.outcome_text(call, week)
        assert "No benchmark is configured" in call.note


class TestSettle:
    def test_repeat_runs_with_the_same_entry_count_once(self):
        rows = [_row("BESI.AS", "2026-09-05T10:00:00+00:00", "negative"),
                _row("BESI.AS", "2026-09-05T19:00:00+00:00", "positive"),
                _row("BESI.AS", "2026-09-06T08:00:00+00:00", "mixed")]  # Sun: same Mon entry
        calls = tr.settle(rows, _loader)
        assert len(calls) == 1 and calls[0].verdict == "mixed"  # the last run counts

    def test_benchmark_follows_the_listing(self):
        calls = tr.settle([_row("BESI.AS", "2026-09-04T19:30:00+00:00"),
                           _row("META", "2026-09-04T19:30:00+00:00")], _loader)
        assert {c.symbol: c.benchmark for c in calls} == {"BESI.AS": "^AEX", "META": "SPY"}

    def test_a_failed_price_fetch_is_a_note_not_a_crash(self):
        def broken(symbol):
            raise OSError("offline")

        calls = tr.settle([_row("BESI.AS", "2026-09-04T19:30:00+00:00")], broken)
        assert calls[0].note == "No price history for the stock."

    def test_summary_by_verdict_and_horizon(self):
        rows = [_row("BESI.AS", "2026-09-04T19:30:00+00:00", "positive"),
                _row("META", "2026-09-04T19:30:00+00:00", "positive")]
        table = tr.summary(tr.settle(rows, _loader)).set_index(["verdict", "horizon"])
        week = table.loc[("positive", "1W")]
        assert week["calls"] == 2 and week["hit_rate"] == pytest.approx(0.5)
        assert list(table.loc["positive"].index) == ["1W", "1M", "3M", "6M"]

    def test_for_symbol_filters_and_handles_none(self):
        rows = [_row("BESI.AS", "2026-09-04T19:30:00+00:00"),
                _row("META", "2026-09-04T19:30:00+00:00")]
        assert [c.symbol for c in tr.for_symbol("besi.as", rows, _loader)] == ["BESI.AS"]
        assert tr.for_symbol("ASML.AS", rows, _loader) == []


class TestPresentation:
    @pytest.fixture
    def calls(self):
        return tr.for_symbol("BESI.AS", [_row("BESI.AS", "2026-09-04T19:30:00+00:00")], _loader)

    def test_report_sections(self, calls):
        from agents import html_report

        report = research_report.ResearchReport(symbol="BESI.AS", verdict="negative",
                                                score=-0.36, confidence=0.59, past_calls=calls)
        page = html_report.render_report(report, "2026-10-28 12:00")
        assert "Past calls on this ticker" in page and "vs ^AEX" in page and "✓" in page
        text = research_report.format_report(report)
        assert "PAST CALLS ON THIS TICKER" in text and ", right" in text

    def test_no_earlier_runs_says_so(self):
        report = research_report.ResearchReport(symbol="X", verdict="mixed", score=0.0,
                                                confidence=0.5, past_calls=[])
        assert "No earlier run of this ticker" in research_report.format_report(report)

    def test_outputs(self, calls, tmp_path):
        csv_path, html_path = tr.write_outputs(calls, tmp_path / "t.csv", tmp_path / "t.html")
        frame = pd.read_csv(csv_path)
        assert frame.loc[0, "benchmark"] == "^AEX" and "excess_1W" in frame.columns
        page = html_path.read_text(encoding="utf-8")
        assert "Track record" in page and "<script" not in page


def test_history_settle_command(monkeypatch, tmp_path, capsys):
    import history

    monkeypatch.setattr(history.signal_snapshots, "load_rows", lambda: [
        _row("BESI.AS", "2026-09-04T19:30:00+00:00"), _row("META", "2026-09-04T19:30:00+00:00",
                                                           "negative")])
    monkeypatch.setattr(history, "load_ohlcv", lambda s: pd.DataFrame({"Close": PRICES[s]}))
    monkeypatch.setattr(tr, "OUTPUT_CSV", tmp_path / "t.csv")
    monkeypatch.setattr(tr, "OUTPUT_HTML", tmp_path / "t.html")
    assert history.main(["settle"]) == 0
    out = capsys.readouterr().out
    assert "2 call(s)" in out and "right 100%" in out
