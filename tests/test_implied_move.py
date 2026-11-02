"""Market move against the move VIX implied (VIX/16): tiers, timing, base rates, output.

Synthetic series only; nothing here touches the network.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agents import charts, flow_evidence, implied_move as im, research_report, signal_snapshots

VIX_FOR_1PCT = 100 * math.sqrt(252) / 100  # VIX level whose implied daily move is 1%


def _series(moves, vix=VIX_FOR_1PCT, start="2026-01-02"):
    """Index closes from a list of moves, and a VIX series on the same sessions."""
    moves = np.asarray(moves, dtype=float)
    days = pd.bdate_range(start, periods=len(moves) + 1)
    close = pd.Series(100.0 * np.concatenate([[1.0], np.cumprod(1.0 + moves)]), index=days)
    vol = pd.Series(np.broadcast_to(np.asarray(vix, dtype=float), len(days)).copy(), index=days)
    return close, vol


def _pattern() -> np.ndarray:
    """200 sessions of +-0.5% with two drops beyond 2x, one notable drop, one big up day."""
    moves = 0.005 * (-1.0) ** np.arange(200)
    moves[[50, 52]] = -0.025
    moves[100] = -0.015
    moves[150] = 0.025
    return moves


class TestRule:
    def test_vix_over_sixteen(self):
        assert im.DIVISOR == pytest.approx(math.sqrt(252)) and round(im.DIVISOR) == 16
        assert im.implied_daily_move(VIX_FOR_1PCT) == pytest.approx(0.01)
        assert im.implied_daily_move(19.45) == pytest.approx(0.01225, abs=1e-5)

    def test_config(self):
        assert (im.INDEX, im.VOL_INDEX, im.NEXT_DAY_VOL_INDEX) == ("^GSPC", "^VIX", "^VIX1D")
        assert (im.NOISE_MAX, im.NOTABLE_MAX) == (1.0, 2.0)

    @pytest.mark.parametrize(("multiple", "expected"), [
        (0.0, im.NOISE), (-1.0, im.NOISE), (1.0001, im.NOTABLE), (-2.0, im.NOTABLE),
        (-2.01, im.BEYOND), (3.5, im.BEYOND), (float("nan"), im.UNKNOWN), (None, im.UNKNOWN)])
    def test_tiers(self, multiple, expected):
        assert im.tier(multiple) == expected

    def test_normal_tail(self):
        assert im.normal_tail(1.0) == pytest.approx(0.158655, abs=1e-6)
        assert im.normal_tail(2.0) == pytest.approx(0.022750, abs=1e-6)


class TestMoveTable:
    def test_implied_move_is_the_previous_close(self):
        close, vol = _series([-0.009, 0.03, -0.005])
        vol.iloc[1] = 2 * VIX_FOR_1PCT  # a VIX jump at the close of the first drop
        table = im.move_table(close, vol)
        assert len(table) == 3
        assert table["implied"].tolist() == pytest.approx([0.01, 0.02, 0.01])
        assert table["multiple"].tolist() == pytest.approx([-0.9, 1.5, -0.5])
        assert table["tier"].tolist() == [im.NOISE, im.NOTABLE, im.NOISE]

    def test_holiday_vix_prints_are_ignored_and_dates_made_naive(self):
        close, vol = _series([0.001] * 5)
        holiday = pd.Timestamp("2026-01-10")  # a Saturday: no index session
        vol = pd.concat([vol, pd.Series([99.0], index=[holiday])]).sort_index()
        close.index = close.index.tz_localize("America/New_York")
        table = im.move_table(close, vol)
        assert holiday not in table.index and table.index.tz is None
        assert table["vol_prev"].max() == pytest.approx(VIX_FOR_1PCT)

    def test_a_missing_vix_close_leaves_the_next_session_out(self):
        close, vol = _series([0.001] * 5)
        vol.iloc[2] = np.nan
        table = im.move_table(close, vol)
        assert close.index[3] not in table.index and len(table) == 4


class TestReading:
    def test_drop_beyond_two_x_changes_the_assumptions(self):
        close, vol = _series([0.002] * 30 + [-0.025])
        mm = im.reading(im.move_table(close, vol))
        assert (mm.tier, mm.assumptions_changed) == (im.BEYOND, True)
        assert "assumptions may have changed" in mm.describe()
        assert mm.recent_beyond_drops == [mm.session] and len(mm.recent) == im.RECENT_SESSIONS

    def test_up_beyond_two_x_does_not(self):
        close, vol = _series([0.002] * 30 + [0.025])
        mm = im.reading(im.move_table(close, vol))
        assert mm.tier == im.BEYOND and not mm.assumptions_changed
        assert "only drops beyond 2x" in mm.tier_text

    def test_noise_and_notable(self):
        close, vol = _series([0.002] * 30 + [-0.015, -0.004])
        mm = im.reading(im.move_table(close, vol))
        assert mm.tier == im.NOISE and not mm.assumptions_changed
        assert mm.recent["tier"].iloc[-2] == im.NOTABLE
        assert mm.recent_text() == "No drop beyond 2x in the last 20 sessions."

    def test_describe_states_the_numbers(self):
        close, vol = _series([0.002] * 5 + [-0.0052])
        text = im.reading(im.move_table(close, vol)).describe()
        assert text.startswith("S&P 500 -0.52% on ") and "±1.00%" in text
        assert f"VIX {VIX_FOR_1PCT:.2f} at the previous close / 15.87" in text
        assert "0.52x, within the implied move: noise" in text

    def test_partial_session_is_flagged(self):
        close, vol = _series([0.002] * 5, start="2026-10-16")
        table = im.move_table(close, vol)
        day = table.index[-1]  # 2026-10-23, a Friday
        during = pd.Timestamp(f"{day.date()} 14:30", tz="America/New_York")
        after = pd.Timestamp(f"{day.date()} 16:30", tz="America/New_York")
        assert im.reading(table, now=during.to_pydatetime()).partial
        assert not im.reading(table, now=after.to_pydatetime()).partial
        assert not im.reading(table, now=datetime(2026, 10, 26, 15, tzinfo=timezone.utc)).partial
        assert "so far on" in im.reading(table, now=during.to_pydatetime()).describe()

    def test_next_day_vix_is_shown_alongside(self):
        close, vol = _series([0.002] * 5 + [-0.012])
        _, vol1d = _series([0.0] * 6, vix=0.8 * VIX_FOR_1PCT)
        mm = im.reading(im.move_table(close, vol), im.move_table(close, vol1d))
        assert mm.next_day_implied == pytest.approx(0.008)
        assert mm.next_day_multiple == pytest.approx(-1.5, rel=1e-6)
        assert mm.tier == im.NOTABLE  # the tiers stay on VIX

    def test_empty_table_raises(self):
        with pytest.raises(ValueError):
            im.reading(pd.DataFrame(columns=["move", "implied"]))


class TestBaseRatesAndStudy:
    @pytest.fixture
    def table(self):
        return im.move_table(*_series(_pattern()))

    def test_base_rates(self, table):
        b = im.base_rates(table)
        assert b.sessions == 200 and b.beyond_drops == 2
        assert b.down_beyond_noise == pytest.approx(3 / 200)
        assert b.down_beyond_notable == pytest.approx(2 / 200)
        assert b.up_beyond_notable == pytest.approx(1 / 200)
        # After the drop at 50 another big move came at 52; after 52 none within 5.
        assert b.follow_after_beyond_drop == pytest.approx(0.5)
        # Sessions with a big move in the next 5: 45..51 and 145..149, of 195 with a window.
        assert b.follow_after_any == pytest.approx(12 / 195)
        text = im.MarketMove(date(2026, 10, 23), -0.005, 15.87, 0.01, -0.5, im.NOISE,
                             base=b).base_text()
        assert "fell beyond 1x its implied move on 1.5% of sessions" in text

    def test_too_short_for_base_rates(self):
        assert im.base_rates(im.move_table(*_series([0.001] * 20))) is None

    def test_follow_through_measures(self):
        table = im.move_table(*_series(0.01 * (-1.0) ** np.arange(60)))
        after = im.follow_through(table)
        ratio = after["realised_to_vix"].dropna()
        assert ratio.iloc[0] == pytest.approx(1.0, abs=0.05)  # +-1% days at a 1% implied move
        expected = table["close"].iloc[im.FORWARD_SESSIONS] / table["close"].iloc[0] - 1.0
        assert after["forward_return"].iloc[0] == pytest.approx(expected)
        assert after["forward_return"].iloc[-im.FORWARD_SESSIONS:].isna().all()

    def test_study_tables(self, table):
        tables = im.study(table)
        freq = tables["frequency"].set_index("multiple")
        assert freq.loc[2.0, "share_down_beyond"] == pytest.approx(2 / 200)
        assert freq.loc[1.0, "normal_curve_per_side"] == pytest.approx(0.158655, abs=1e-6)
        assert tables["by_vix"]["vix_level"].tolist() == ["15-20"]  # VIX 15.87 throughout
        follow = tables["follow"].set_index("day")
        assert follow.loc["all sessions", "sessions"] == 200 - im.FORWARD_SESSIONS
        assert follow.loc["down 2x-3x", "sessions"] == 2

    def test_study_outputs(self, table, tmp_path):
        csv_path, html_path = im.write_study(im.study(table), im.base_rates(table),
                                             tmp_path / "s.csv", tmp_path / "s.html")
        text = csv_path.read_text(encoding="utf-8")
        assert "frequency" in text and "by_vix" in text and "follow" in text
        page = html_path.read_text(encoding="utf-8")
        assert "Implied move study" in page and "<script" not in page


class TestPresentation:
    @pytest.fixture
    def mm(self):
        close, vol = _series(list(0.004 * (-1.0) ** np.arange(30)) + [-0.025])
        return im.reading(im.move_table(close, vol),
                          base=im.base_rates(im.move_table(*_series(_pattern()))))

    def test_chart(self, mm):
        svg = charts.implied_move_chart(mm.recent, im.NOISE_MAX, im.NOTABLE_MAX, True)
        assert svg.count('class="im-bar') == len(mm.recent) == im.RECENT_SESSIONS
        assert 'class="im-bar beyond down partial"' in svg and "<script" not in svg
        assert svg.count('class="im-band1"') == len(mm.recent)
        assert "within 1x implied (noise)" in svg and "(so far)" in svg

    def test_html_section(self, mm):
        from agents import html_report

        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.0,
                                                confidence=0.5, market_move=mm)
        page = html_report.render_report(report, "2026-10-25 12:00")
        assert "Market move vs implied (VIX/16)" in page
        assert "ASSUMPTIONS MAY HAVE CHANGED" in page and "im-chart" in page
        assert "Shown, not scored." in page

    def test_terminal_section(self, mm):
        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.0,
                                                confidence=0.5, market_move=mm)
        text = research_report.format_report(report)
        assert "MARKET MOVE VS IMPLIED (VIX/16) -> BEYOND" in text
        assert "1 drop(s) beyond 2x in the last 20 sessions" in text

    def test_evidence_line(self, mm):
        line = flow_evidence.market_move_line(mm)
        assert line.startswith("S&P 500 -2.50% on ") and "2.50x, beyond" in line
        assert line.endswith("(shown, not scored)")

    def test_snapshot_fields(self, mm):
        report = research_report.ResearchReport(symbol="JPM", verdict="mixed", score=0.0,
                                                confidence=0.5, market_move=mm)
        row = signal_snapshots.snapshot_row(report, datetime(2026, 10, 25, tzinfo=timezone.utc))
        assert row["market_move_tier"] == im.BEYOND and row["market_move_partial"] == "false"
        assert float(row["market_move_multiple"]) == mm.multiple
        assert row["market_move_session"] == mm.session.isoformat()


def test_analyze_offline(monkeypatch):
    close, vol = _series([0.002] * 40 + [-0.013])
    _, vol1d = _series([0.0] * 41, vix=VIX_FOR_1PCT)
    long_close, long_vol = _series(_pattern(), start="2000-01-03")
    frames = {im.INDEX: close, im.VOL_INDEX: vol, im.NEXT_DAY_VOL_INDEX: vol1d}
    monkeypatch.setattr(im, "fetch_price_history",
                        lambda symbol, period=None: pd.DataFrame({"Close": frames[symbol]}))
    monkeypatch.setattr(im, "load_ohlcv", lambda symbol: pd.DataFrame(
        {"Close": {im.INDEX: long_close, im.VOL_INDEX: long_vol}[symbol]}))
    mm = im.analyze(now=datetime(2026, 12, 1, tzinfo=timezone.utc))
    assert mm.tier == im.NOTABLE and mm.next_day_implied == pytest.approx(0.01)
    assert mm.base is not None and mm.base.sessions == 200


def test_analyze_without_vix1d_or_history(monkeypatch):
    close, vol = _series([0.002] * 40 + [-0.004])

    def fetch(symbol, period=None):
        if symbol == im.NEXT_DAY_VOL_INDEX:
            raise ValueError("no data")
        return pd.DataFrame({"Close": close if symbol == im.INDEX else vol})

    def no_history(symbol):
        raise OSError("offline")

    monkeypatch.setattr(im, "fetch_price_history", fetch)
    monkeypatch.setattr(im, "load_ohlcv", no_history)
    mm = im.analyze()
    assert mm.tier == im.NOISE and mm.next_day_implied is None and mm.base is None


def test_history_study_command(monkeypatch, tmp_path, capsys):
    import history

    close, vol = _series(_pattern(), start="2000-01-03")
    monkeypatch.setattr(history, "load_ohlcv", lambda symbol: pd.DataFrame(
        {"Close": close if symbol == im.INDEX else vol}))
    monkeypatch.setattr(im, "OUTPUT_CSV", tmp_path / "s.csv")
    monkeypatch.setattr(im, "OUTPUT_HTML", tmp_path / "s.html")
    assert history.main(["implied-move-study"]) == 0
    assert "Implied move study" in capsys.readouterr().out
    assert (tmp_path / "s.html").exists()
