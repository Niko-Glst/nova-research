"""Point-in-time backfill and the flow backtest, on synthetic data. No network."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents import flow_analyst as fl
from agents import flow_history as fh
from agents import snapshot_store

REC = "2026-10-22T12:00:00+00:00"


def _ohlc(n: int = 700, seed: int = 0, start: str = "2018-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.bdate_range(start, periods=n))


def _facts():
    def shares(accn, filed, values):
        return [{"end": e, "val": v, "accn": accn, "form": "10-K", "fp": "FY", "filed": filed}
                for e, v in values]

    return {"facts": {"us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares":
        shares("A1", "2019-02-15", [("2018-12-31", 110.0), ("2017-12-31", 100.0)])
        + shares("A2", "2020-02-14", [("2019-12-31", 100.0), ("2018-12-31", 110.0)])}}}}}


class TestBackfill:
    def test_price_rows_use_only_the_past(self):
        ohlc = _ohlc()
        dates = fh.scoring_dates(ohlc.index, pd.Timestamp("2019-03-01"))
        first = fh.price_backfill("X", ohlc, dates, REC)
        changed = ohlc.copy()
        cut = dates[5]
        changed.loc[changed.index > cut] *= 1.5
        second = fh.price_backfill("X", changed, dates, REC)
        early = first[pd.to_datetime(first["date"]) <= cut]
        pd.testing.assert_frame_equal(early.reset_index(drop=True),
                                      second[pd.to_datetime(second["date"]) <= cut]
                                      .reset_index(drop=True))
        assert set(first["source"]) == {snapshot_store.PRICE_BACKFILL}
        assert {"stock_vol_z", "stock_vol_ratio", "vol_estimator"} <= set(first["field"])

    def test_sec_rows_are_dated_by_filing(self):
        rows = fh.sec_backfill("X", _facts(), REC)
        ni = rows[rows["field"] == "net_issuance"]
        assert list(ni["date"]) == ["2019-02-15", "2020-02-14"]
        assert list(ni["filed_date"]) == ["2019-02-15", "2020-02-14"]
        assert set(rows[rows["field"] == "buyback_flag"]["value"]) == {fl.CONFIRMED_ZERO}
        assert set(rows["source"]) == {snapshot_store.SEC_PIT}


class TestPanel:
    def _setup(self):
        bench = _ohlc(seed=1)
        stock = _ohlc(seed=2)
        dates = fh.scoring_dates(bench.index, pd.Timestamp("2019-01-01"))
        long = pd.concat([
            fh.market_backfill(bench["Close"], dates, REC, fh.BENCHMARK),
            fh.price_backfill("X", stock, dates, REC),
            fh.sec_backfill("X", _facts(), REC),
        ], ignore_index=True)
        panel = fh.build_panel(long, {"X": stock["Close"]}, bench["Close"], dates)
        return panel.set_index("date"), stock, bench

    def test_a_10k_counts_only_after_it_was_filed(self):
        panel, _, _ = self._setup()
        before = panel.loc[pd.Timestamp("2019-01-31")]
        after = panel.loc[pd.Timestamp("2019-02-28")]
        assert math.isnan(before["issuance"])  # A1 filed 2019-02-15
        assert after["issuance"] == pytest.approx(fl.issuance_score(math.log(1.1)))
        # A2 (filed 2020-02-14): shares fell, no repurchase fact -> data_conflict, dropped.
        assert math.isnan(panel.loc[pd.Timestamp("2020-02-28")]["buyback"])

    def test_a_stale_filing_is_not_current(self):
        panel, _, _ = self._setup()
        late = panel.loc[pd.Timestamp("2020-01-31")]  # A1 is 350 days old: still current
        assert not math.isnan(late["issuance"])

    def test_forward_excess_return(self):
        panel, stock, bench = self._setup()
        day = pd.Timestamp("2019-06-28")
        end = day + pd.DateOffset(months=1)
        expected = (stock["Close"].loc[:end].iloc[-1] / stock["Close"].loc[:day].iloc[-1]
                    - bench["Close"].loc[:end].iloc[-1] / bench["Close"].loc[:day].iloc[-1])
        assert panel.loc[day]["fwd_1m"] == pytest.approx(expected)


class TestStatistics:
    def _panel(self, signal: float, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for d in pd.date_range("2015-01-31", periods=60, freq="ME"):
            score = rng.normal(size=20)
            for i in range(20):
                fwd = {f"fwd_{h}m": signal * score[i] + rng.normal() for h in fh.HORIZONS}
                rows.append({"date": d, "ticker": f"T{i}", "score": score[i], "vol": score[i],
                             "issuance": np.nan, "buyback": np.nan, "market": 1.0, **fwd})
        return pd.DataFrame(rows)

    def test_planted_signal_is_found(self):
        rows = {(r.component, r.horizon_months): r for r in
                fh.information_coefficients(self._panel(0.8))}
        best = rows[("score", 1)]
        assert best.mean_ic > 0.4 and best.ic_t > 5 and best.spread > 0
        assert rows[("issuance", 1)].n_dates == 0  # all NaN: no IC, not zero

    def test_noise_gives_no_signal(self):
        row = next(r for r in fh.information_coefficients(self._panel(0.0, seed=3))
                   if r.component == "score" and r.horizon_months == 1)
        assert abs(row.mean_ic) < 0.06 and abs(row.ic_t) < 2.5

    def test_newey_west_lag_zero_is_the_plain_t(self):
        x = np.random.default_rng(4).normal(0.1, 1, 200)
        plain = x.mean() / (x.std(ddof=0) / math.sqrt(200))
        assert fh.newey_west_t(x, 0) == pytest.approx(plain)

    def test_outputs_are_written(self, tmp_path):
        rows = fh.information_coefficients(self._panel(0.5))
        market = pd.DataFrame([{"horizon_months": 1, "market_reading": 1.0, "months": 10,
                                "mean_benchmark_return": 0.01}])
        meta = {"model_version": "2.0.0", "universe": "u", "tickers": 20, "benchmark": "SPY",
                "first": "2015-01-31", "last": "2019-12-31", "notes": ["n"]}
        csv_path, html_path = fh.write_outputs(rows, market, meta, tmp_path / "b.csv",
                                               tmp_path / "b.html")
        assert "cross_section_ic" in csv_path.read_text(encoding="utf-8")
        assert "Information coefficient" in html_path.read_text(encoding="utf-8")
