"""Breadth backtest: point-in-time regimes, forward outcomes, statistics and outputs.

Synthetic prices only; nothing here touches the network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents import breadth_analyst as ba
from agents import breadth_history as bh

DAYS = pd.bdate_range("2004-01-02", "2007-12-31")


def _wave() -> pd.Series:
    """RSP/SPY swinging through two broadening and narrowing waves."""
    t = np.arange(len(DAYS))
    return pd.Series(0.30 * np.exp(0.06 * np.sin(2 * np.pi * t / 500)), index=DAYS)


def _growth(rate: float) -> pd.Series:
    return pd.Series(100.0 * np.exp(rate * np.arange(len(DAYS))), index=DAYS)


class TestPanel:
    def test_regimes_are_point_in_time(self):
        ratio = _wave()
        dates = bh.scoring_dates(ratio.index, pd.Timestamp("2005-03-01"))
        panel = bh.regimes(ratio, dates)
        day = panel.index[3]
        assert panel.loc[day, "state"] == ba.regime_from_ratio(ratio.loc[:day]).state
        changed = ratio.copy()
        changed.loc[changed.index > day] *= 1.5  # the future must not matter
        assert bh.regimes(changed, pd.DatetimeIndex([day])).loc[day, "regime"] == \
            panel.loc[day, "regime"]

    def test_dates_before_a_full_range_are_skipped(self):
        ratio = _wave()
        panel = bh.regimes(ratio, bh.scoring_dates(ratio.index, pd.Timestamp("2004-01-01")))
        assert panel.index.min() >= ratio.index[ba.RANGE_DAYS - 1]

    def test_forward_log_change(self):
        series = _growth(0.001)
        day = pd.Timestamp("2005-06-30")
        expected = math.log(series.loc[:day + pd.DateOffset(months=1)].iloc[-1]
                            / series.loc[day])
        assert bh.forward_log_change(series, day, 1) == pytest.approx(expected)
        assert math.isnan(bh.forward_log_change(series, pd.Timestamp("2007-11-30"), 3))

    def test_sector_outcome_is_breadth_basket_minus_leaders(self):
        ratio = _wave()
        breadth = {"XLV": _growth(0.0010), "XLF": _growth(0.0006)}
        leaders = {"XLK": _growth(0.0002)}
        panel = bh.build_panel(ratio, breadth, leaders,
                               bh.scoring_dates(ratio.index, pd.Timestamp("2005-06-01")))
        day = panel.index[0]
        expected = (np.mean([bh.forward_log_change(s, day, 1) for s in breadth.values()])
                    - bh.forward_log_change(leaders["XLK"], day, 1))
        assert panel.loc[day, "sectors_1m"] == pytest.approx(expected)
        assert {f"ratio_{h}m" for h in bh.HORIZONS} <= set(panel.columns)


def _panel(regime, state, outcome) -> pd.DataFrame:
    frame = pd.DataFrame({"regime": regime, "state": state},
                         index=pd.date_range("2010-01-31", periods=len(regime), freq="ME"))
    for h in bh.HORIZONS:
        frame[f"ratio_{h}m"] = outcome
        frame[f"sectors_{h}m"] = outcome
    return frame


def _call(rows, label="all", outcome="ratio", h=1):
    return next(r for r in rows if (r.section, r.label, r.outcome, r.horizon_months)
                == ("call", label, outcome, h))


class TestSummary:
    def test_right_calls_score_positive(self):
        regime = [1.0, -1.0] * 30
        state = ["trend_broadening", "trend_narrowing"] * 30
        outcome = [0.01 + 0.002 * (i % 3) if r > 0 else -0.01 for i, r in enumerate(regime)]
        rows = bh.summarize(_panel(regime, state, outcome))
        call = _call(rows)
        assert call.months == 60 and call.hit_rate == 1.0 and call.mean > 0.01
        assert _call(rows, "trend").months == 60 and _call(rows, "zone").months == 0

    def test_a_call_that_only_follows_the_drift_nets_to_zero(self):
        """Always 'narrowing' while the ratio always falls: right, but by drift alone."""
        rows = bh.summarize(_panel([-1.0] * 40, ["trend_narrowing"] * 40,
                                   [-0.01 - 0.001 * (i % 4) for i in range(40)]))
        call = _call(rows)
        assert call.mean > 0 and call.hit_rate == 1.0
        assert call.net_of_drift == pytest.approx(0.0, abs=1e-12)

    def test_waiting_months_are_not_calls(self):
        rows = bh.summarize(_panel([0.0, 1.0] * 20, ["bottom_waiting", "bottom_turn"] * 20,
                                   [0.02, 0.01] * 20))
        assert _call(rows).months == 20 and _call(rows, "zone").months == 20
        waiting = next(r for r in rows if r.section == "state" and r.label == "bottom_waiting"
                       and r.outcome == "ratio" and r.horizon_months == 1)
        assert waiting.months == 20 and math.isnan(waiting.hit_rate)

    def test_outputs_are_written(self, tmp_path):
        rows = bh.summarize(_panel([1.0, -1.0] * 10, ["bottom_turn", "top_turn"] * 10,
                                   [0.01, -0.01] * 10))
        csv_path, html_path = bh.write_outputs(
            rows, {"model_version": "2.0.0", "weight": 0.8, "first": "2010-01-31",
                   "last": "2011-08-31", "notes": ["a note"]},
            tmp_path / "b.csv", tmp_path / "b.html")
        table = pd.read_csv(csv_path)
        assert set(table["section"]) == {"regime", "state", "call"}
        page = html_path.read_text(encoding="utf-8")
        assert "Breadth backtest" in page and "Net of drift" in page and "a note" in page
        assert "<script" not in page


def test_history_command_runs_offline(monkeypatch, tmp_path, capsys):
    import history

    prices = {ba.EQUAL_WEIGHT: _wave() * 100.0, ba.CAP_WEIGHT: pd.Series(100.0, index=DAYS)}
    for i, symbol in enumerate(bh.BREADTH_ETFS + bh.LEADER_ETFS):
        prices[symbol] = _growth(0.0002 * (i + 1))
    monkeypatch.setattr(history, "load_ohlcv",
                        lambda symbol: pd.DataFrame({"Close": prices[symbol]}))
    monkeypatch.setattr(bh, "OUTPUT_CSV", tmp_path / "breadth.csv")
    monkeypatch.setattr(bh, "OUTPUT_HTML", tmp_path / "breadth.html")
    assert history.main(["breadth-backtest"]) == 0
    out = capsys.readouterr().out
    assert "Breadth backtest" in out and "month-ends" in out
    assert (tmp_path / "breadth.csv").exists() and (tmp_path / "breadth.html").exists()
