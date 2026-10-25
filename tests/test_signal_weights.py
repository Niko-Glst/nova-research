"""Tests for the signal register, the weight fitters and the walk-forward engine.

The ones that matter most guard against look-ahead: a signal on day t must not
change when later prices change, fitted weights must not depend on the test
period, and an after-close earnings report must not count until the next day.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.strategy_agent import moving_average_crossover
from backtests import market_data, signal_weights as sw, signals, walk_forward as wf
from backtests.backtest_engine import run_backtest


def _ohlcv(n=900, seed=0, drift=0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    index = pd.bdate_range("2015-01-01", periods=n)
    high, low = close * 1.01, close * 0.99
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": volume}, index=index)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


class TestSignals:
    def test_no_lookahead_in_any_technical_signal(self):
        data = _ohlcv()
        cut = 600
        base = signals.compute_all(signals.SignalInputs("X", data))
        changed = data.copy()
        changed.iloc[cut + 1:, :4] *= 1.5  # rewrite the future
        after = signals.compute_all(signals.SignalInputs("X", changed))
        pd.testing.assert_frame_equal(base.iloc[: cut + 1], after.iloc[: cut + 1])

    def test_all_values_in_range(self):
        frame = signals.compute_all(signals.SignalInputs("X", _ohlcv()))
        values = frame.to_numpy()
        assert np.nanmin(values) >= -1 and np.nanmax(values) <= 1

    def test_trend_sign_is_the_crossover(self):
        data = _ohlcv()
        trend = signals.REGISTRY["trend"].compute(signals.SignalInputs("X", data))
        close = data["Close"]
        cross = close.rolling(50).mean() > close.rolling(200).mean()
        valid = trend.notna()
        assert ((trend[valid] > 0) == cross[valid]).all()

    def test_register_a_new_parameter(self):
        name = "test_five_day_return"
        signals.register(signals.Signal(
            name, "custom", "five-day return",
            lambda i: (i.ohlcv["Close"].pct_change(5) / 0.05).clip(-1, 1)))
        try:
            frame = signals.compute_all(signals.SignalInputs("X", _ohlcv()), [name, "trend"])
            assert list(frame.columns) == [name, "trend"]
            with pytest.raises(ValueError, match="already registered"):
                signals.register(signals.REGISTRY[name])
        finally:
            del signals.REGISTRY[name]

    def test_signals_without_history_are_refused(self):
        assert {s.name for s in signals.unavailable()} >= {"news_sentiment", "insider_flow"}
        with pytest.raises(ValueError, match="no point-in-time history"):
            signals.compute_all(signals.SignalInputs("X", _ohlcv()), ["news_sentiment"])

    def test_bad_group(self):
        with pytest.raises(ValueError, match="group"):
            signals.Signal("x", "astrology", "", lambda i: i.ohlcv["Close"])


class TestEarnings:
    def _days(self):
        return pd.bdate_range("2024-01-01", periods=40)

    def test_after_close_counts_from_the_next_day(self):
        announced = pd.Series(pd.to_datetime([
            "2024-01-10 16:05", "2024-01-17 07:30", "2024-01-20 16:30",  # Sat after close
        ]).tz_localize("America/New_York"))
        got = market_data.available_from(announced, self._days())
        assert [d.date().isoformat() for d in got] == ["2024-01-11", "2024-01-17", "2024-01-22"]

    def test_surprise_appears_on_the_right_close_and_decays(self):
        days = self._days()
        ohlcv = pd.DataFrame({"Close": np.linspace(100, 110, len(days))}, index=days)
        earnings = pd.DataFrame({
            "announced": pd.to_datetime(["2024-01-10 16:05"]).tz_localize("America/New_York"),
            "eps_estimate": [1.0], "eps_reported": [1.1], "surprise_pct": [10.0],
        })
        s = signals.REGISTRY["earnings_surprise"].compute(signals.SignalInputs("X", ohlcv, earnings))
        assert math.isnan(s.loc["2024-01-10"])  # not yet public at that close
        assert s.loc["2024-01-11"] == pytest.approx(1.0)  # 10% surprise = full reading
        later = days[days.get_loc(pd.Timestamp("2024-01-11")) + signals.SURPRISE_HALF_LIFE_DAYS] \
            if len(days) > 31 + 8 else None
        if later is not None:
            assert s.loc[later] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Portfolio arithmetic and fitters
# --------------------------------------------------------------------------


class TestPortfolio:
    def test_fees_only_on_changes_and_carry_is_free(self):
        positions = np.array([[1.0], [1.0], [0.0]])
        returns = np.array([[0.01], [0.02], [0.03]])
        out = sw.portfolio_returns(positions, returns, previous=np.array([1.0]))
        fee = sw.FEE_BPS / 10_000
        assert out.daily.tolist() == pytest.approx([0.01, 0.02, -fee])
        fresh = sw.portfolio_returns(positions, returns)
        assert fresh.daily[0] == pytest.approx(0.01 - fee)

    def test_batched_sharpes_match_one_at_a_time(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, size=(400, 5, 3))
        r = rng.normal(0.0005, 0.01, size=(400, 5))
        w = rng.uniform(-1, 1, size=(3, 4))
        w /= np.abs(w).sum(axis=0)
        batch = sw._batch_sharpes(x, r, w, 0.05)
        for j in range(4):
            single = sw.sharpe(sw.evaluate(x, r, sw.Weights("t", w[:, j], 0.05, ("a", "b", "c"))).daily)
            assert batch[j] == pytest.approx(single, rel=1e-4)

    def test_crossover_matches_the_existing_backtest_engine(self):
        """The weighted framework with trend only, theta 0, is the pipeline's rule."""
        data = _ohlcv(1200, seed=3)
        frame = signals.compute_all(signals.SignalInputs("X", data), ["trend"])
        panel = wf.build_panel({"X": frame}, {"X": data["Close"]})
        rule = sw.fixed(panel.names, {"trend": 1.0})
        ours = sw.evaluate(panel.x, panel.returns, rule).daily

        sig = moving_average_crossover(data)
        engine = run_backtest(data, sig.entries, sig.exits, fees_bps=sw.FEE_BPS)
        engine_daily = engine.equity_curve.pct_change().to_numpy()
        # The engine waits for the first crossover; compare from there on.
        first = int(np.argmax(sig.entries.to_numpy())) + 2
        assert np.allclose(ours[first:], engine_daily[first:], atol=1e-12)


class TestFitters:
    def _panel(self, t=1500, n=6, seed=2):
        """Signal 'a' predicts tomorrow's return; 'b' and 'c' are noise."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, size=(t, n, 3))
        r = 0.004 * x[:, :, 0] + rng.normal(0.0003, 0.01, size=(t, n))
        return x, r, ("a", "b", "c")

    def test_ridge_finds_the_predictive_signal(self):
        x, r, names = self._panel()
        w = sw.fit_ridge(x, r, names)
        assert np.argmax(np.abs(w.weights)) == 0 and w.weights[0] > 0

    def test_ridge_is_the_closed_form(self):
        x, r, names = self._panel(t=300, n=3)
        k = 3
        scale = np.nanstd(r, axis=0, ddof=1)
        y = (r / scale).reshape(-1)
        rows = x.reshape(-1, k)
        xc, yc = rows - rows.mean(0), y - y.mean()
        lam = sw.RIDGE_PENALTY * np.trace(xc.T @ xc) / k
        direct = np.linalg.inv(xc.T @ xc + lam * np.eye(k)) @ xc.T @ yc
        got = sw.fit_ridge(x, r, names).weights
        assert got == pytest.approx(direct / np.abs(direct).sum())

    def test_max_sharpe_and_search_favour_the_predictive_signal(self):
        x, r, names = self._panel()
        for fitter in (sw.fit_max_sharpe, lambda *a: sw.fit_search(*a, trials=300)):
            w = fitter(x, r, names)
            assert np.argmax(np.abs(w.weights)) == 0, w

    def test_signal_returns_shape(self):
        x, r, _ = self._panel(t=50, n=4)
        assert sw.signal_returns(x, r).shape == (50, 3)


# --------------------------------------------------------------------------
# Walk-forward and statistics
# --------------------------------------------------------------------------


class TestStatistics:
    def test_norm_ppf_matches_published_quantiles(self):
        assert wf.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
        assert wf.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        assert wf.norm_ppf(0.001) == pytest.approx(-3.090232, abs=1e-6)

    def test_deflated_sharpe_punishes_many_noise_tries(self):
        rng = np.random.default_rng(4)
        noise = rng.normal(0, 0.01, size=(1000, 500))
        sharpes = noise.mean(0) / noise.std(0, ddof=1) * np.sqrt(252)
        best = noise[:, int(np.argmax(sharpes))]
        assert wf.deflated_sharpe(best, sharpes) < 0.95

    def test_deflated_sharpe_accepts_a_real_edge(self):
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 0.01, size=(1000, 50))
        real = rng.normal(0.003, 0.01, 1000)  # annual Sharpe ~ 4.8
        sharpes = np.append(noise.mean(0) / noise.std(0, ddof=1), real.mean() / real.std()) * np.sqrt(252)
        assert wf.deflated_sharpe(real, sharpes) > 0.95

    def test_sharpe_difference_test(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.0005, 0.01, 2000)
        b = 0.5 * a + rng.normal(0.0002, 0.01, 2000)
        ab, ba = wf.sharpe_difference_test(a, b), wf.sharpe_difference_test(b, a)
        assert ab.z == pytest.approx(-ba.z) and ab.p_value == pytest.approx(ba.p_value)
        better = wf.sharpe_difference_test(a + 0.002, a - 0.002)
        assert better.difference > 0 and better.p_value < 0.01


class TestWalkForward:
    def _inputs(self, n_stocks=4, t=1100):
        sigs, closes = {}, {}
        for i in range(n_stocks):
            data = _ohlcv(t, seed=10 + i)
            sigs[f"S{i}"] = signals.compute_all(signals.SignalInputs(f"S{i}", data))
            closes[f"S{i}"] = data["Close"]
        return sigs, closes

    def test_runs_and_stitches_out_of_sample(self):
        sigs, closes = self._inputs()
        panel = wf.build_panel(sigs, closes)
        result = wf.walk_forward(panel, ("ridge", "equal"), train_days=500, step_days=100)
        oos = len(panel.dates) - 500
        assert len(result.methods["ridge"].daily) == oos
        assert len(result.buy_hold.daily) == oos and result.crossover is not None
        assert result.refits[0].test_start == panel.dates[500]

    def test_weights_do_not_see_the_test_period(self):
        sigs, closes = self._inputs()
        panel = wf.build_panel(sigs, closes)
        first = wf.walk_forward(panel, ("ridge",), 500, 100).refits[0].weights["ridge"]
        tampered_returns = panel.returns.copy()
        tampered_returns[500:] *= -3  # rewrite everything after the first training window
        tampered = wf.Panel(panel.dates, panel.symbols, panel.names, panel.x, tampered_returns,
                            panel.coverage)
        again = wf.walk_forward(tampered, ("ridge",), 500, 100).refits[0].weights["ridge"]
        assert again.weights == pytest.approx(first.weights)

    def test_signals_are_lagged_one_day(self):
        sigs, closes = self._inputs(2, 400)
        panel = wf.build_panel(sigs, closes)
        s0 = sigs["S0"]["trend"]
        day = panel.dates[300]
        previous = s0.index[s0.index.get_loc(day) - 1]
        assert panel.x[300, 0, panel.names.index("trend")] == pytest.approx(s0.loc[previous])

    def test_too_short_history(self):
        sigs, closes = self._inputs(2, 300)
        with pytest.raises(ValueError, match="need more than"):
            wf.walk_forward(wf.build_panel(sigs, closes), ("equal",), 500, 100)


class TestSnapshots:
    def test_append_only_with_one_header(self, tmp_path):
        from types import SimpleNamespace

        from agents import signal_snapshots

        report = SimpleNamespace(symbol="AAPL", verdict="mixed", score=0.07, confidence=1.0,
                                 technical=None, fundamental=None, sentiment=None,
                                 insider=SimpleNamespace(signal="bearish", buy_count=0,
                                                         sell_count=16, net_value=-1.8e8))
        path = tmp_path / "signals.csv"
        signal_snapshots.record(report, path)
        signal_snapshots.record(report, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3 and lines[0].startswith("recorded_at,symbol")
        frame = pd.read_csv(path)
        assert frame["insider_sells"].tolist() == [16, 16]
