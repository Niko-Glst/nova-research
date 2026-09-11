"""Tests for the backtest engine, Monte Carlo, and the OLS implementation.

The statistical functions are checked against values from published tables, not
against another implementation of the same idea -- an error shared between two
of my own implementations would cancel out and look like agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests import monte_carlo as mc
from backtests import regression as reg
from backtests.backtest_engine import run_backtest


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=index)


def _signals(index, entry_at: list[int], exit_at: list[int]):
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    for i in entry_at:
        entries.iloc[i] = True
    for i in exit_at:
        exits.iloc[i] = True
    return entries, exits


class TestBacktestMechanics:
    def test_always_in_matches_buy_and_hold(self):
        frame = _frame(list(np.linspace(100, 200, 300)))
        entries, exits = _signals(frame.index, [0], [])
        result = run_backtest(frame, entries, exits, symbol="T")
        assert result.total_return_pct == pytest.approx(
            result.buy_and_hold_return_pct, abs=0.5
        )

    def test_never_in_earns_nothing(self):
        frame = _frame(list(np.linspace(100, 200, 100)))
        entries, exits = _signals(frame.index, [], [])
        result = run_backtest(frame, entries, exits, symbol="T")
        assert result.total_return_pct == pytest.approx(0.0)
        assert result.num_trades == 0
        assert result.time_in_market_pct == 0.0

    def test_fees_reduce_returns(self):
        frame = _frame(list(np.linspace(100, 200, 200)))
        entries, exits = _signals(frame.index, [0, 100], [50, 150])
        free = run_backtest(frame, entries, exits, fees_bps=0)
        charged = run_backtest(frame, entries, exits, fees_bps=100)
        assert charged.total_return_pct < free.total_return_pct

    def test_signals_are_shifted_by_one_bar(self):
        """A signal from a close cannot be traded at that same close.

        Entering on the last bar must therefore earn nothing, not the final move.
        """
        frame = _frame([100.0] * 10 + [200.0])
        entries, exits = _signals(frame.index, [len(frame) - 1], [])
        result = run_backtest(frame, entries, exits)
        assert result.total_return_pct == pytest.approx(0.0)

    def test_unshifted_signals_can_be_requested(self):
        frame = _frame(list(np.linspace(100, 200, 100)))
        entries, exits = _signals(frame.index, [0], [])
        shifted = run_backtest(frame, entries, exits, shift_signals=True)
        raw = run_backtest(frame, entries, exits, shift_signals=False)
        assert raw.time_in_market_pct > shifted.time_in_market_pct

    def test_exit_while_flat_is_ignored(self):
        frame = _frame(list(np.linspace(100, 150, 60)))
        entries, exits = _signals(frame.index, [30], [5, 10])
        result = run_backtest(frame, entries, exits)
        assert result.num_trades == 1

    def test_repeat_entries_do_not_stack(self):
        frame = _frame(list(np.linspace(100, 150, 60)))
        entries, exits = _signals(frame.index, [5, 10, 15, 20], [])
        result = run_backtest(frame, entries, exits)
        assert result.num_trades == 1

    def test_drawdown_is_negative_on_a_decline(self):
        frame = _frame([100, 120, 140, 90, 95, 110])
        entries, exits = _signals(frame.index, [0], [])
        result = run_backtest(frame, entries, exits)
        assert result.max_drawdown_pct < 0

    def test_open_position_is_closed_at_the_last_bar(self):
        frame = _frame(list(np.linspace(100, 150, 60)))
        entries, exits = _signals(frame.index, [10], [])
        result = run_backtest(frame, entries, exits)
        assert result.num_trades == 1
        assert result.trades[0].exit_index == len(frame) - 1

    def test_equity_curve_aligns_with_price_index(self):
        frame = _frame(list(np.linspace(100, 150, 40)))
        entries, exits = _signals(frame.index, [0], [])
        result = run_backtest(frame, entries, exits)
        assert result.equity_curve is not None
        assert list(result.equity_curve.index) == list(frame.index)

    def test_empty_history_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            run_backtest(pd.DataFrame({"Close": []}), pd.Series([]), pd.Series([]))

    def test_missing_close_column_is_rejected(self):
        frame = pd.DataFrame({"Open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Close"):
            run_backtest(frame, pd.Series([True, False]), pd.Series([False, False]))


class TestMonteCarlo:
    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(7)
        return rng.normal(0.0005, 0.015, 500)

    def test_resample_shape(self, returns):
        paths = mc.resample_returns(returns, 200, 60, seed=1)
        assert paths.shape == (200, 60)

    def test_seed_makes_it_reproducible(self, returns):
        a = mc.resample_returns(returns, 100, 30, seed=42)
        b = mc.resample_returns(returns, 100, 30, seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self, returns):
        a = mc.resample_returns(returns, 100, 30, seed=1)
        b = mc.resample_returns(returns, 100, 30, seed=2)
        assert not np.array_equal(a, b)

    def test_resampled_values_come_from_the_original(self, returns):
        paths = mc.resample_returns(returns, 50, 20, seed=3)
        assert np.isin(paths, returns).all()

    def test_block_method_preserves_contiguity(self):
        """Consecutive values in a block must be consecutive in the source."""
        source = np.arange(100, dtype=float)
        paths = mc.resample_returns(source, 20, 21, method="block", block_size=21, seed=5)
        # Within a block, each step increases by exactly 1 in the source ordering.
        diffs = np.diff(paths[0][:21])
        assert (diffs == 1.0).all()

    def test_iid_method_is_available(self, returns):
        paths = mc.resample_returns(returns, 50, 20, method="iid", seed=1)
        assert paths.shape == (50, 20)

    def test_unknown_method_is_rejected(self, returns):
        with pytest.raises(ValueError, match="Unknown method"):
            mc.resample_returns(returns, 10, 10, method="magic")

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            mc.resample_returns(np.array([0.01]), 10, 10)

    def test_percentiles_are_ordered(self, returns):
        result = mc.run_monte_carlo("T", returns, num_simulations=500, seed=1)
        p = result.return_percentiles
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]

    def test_probability_of_loss_is_a_probability(self, returns):
        result = mc.run_monte_carlo("T", returns, num_simulations=500, seed=1)
        assert 0.0 <= result.probability_of_loss <= 1.0

    def test_cvar_is_at_most_var(self, returns):
        """The mean of the worst tail cannot exceed the tail's boundary."""
        result = mc.run_monte_carlo("T", returns, num_simulations=1000, seed=1)
        assert result.conditional_var_5pct <= result.value_at_risk_5pct

    def test_positive_drift_yields_positive_median(self):
        rng = np.random.default_rng(11)
        strong = rng.normal(0.002, 0.01, 400)
        result = mc.run_monte_carlo("T", strong, num_simulations=500, seed=2)
        assert result.median_return_pct > 0

    def test_observed_percentile_is_located(self, returns):
        result = mc.run_monte_carlo(
            "T", returns, num_simulations=500, observed_return_pct=1000.0, seed=1
        )
        # An implausibly good result sits at the very top of the distribution.
        assert result.observed_percentile is not None
        assert result.observed_percentile > 95

    def test_summary_mentions_a_lucky_draw(self, returns):
        result = mc.run_monte_carlo(
            "T", returns, num_simulations=500, observed_return_pct=1e6, seed=1
        )
        assert "favourable draw" in result.summary()


class TestOlsAgainstPublishedValues:
    """p-values checked against standard statistical tables."""

    @pytest.mark.parametrize(
        ("t", "df", "expected"),
        [(2.086, 20, 0.050), (3.169, 10, 0.010), (1.960, 1000, 0.0502), (0.0, 10, 1.0)],
    )
    def test_t_distribution_matches_tables(self, t, df, expected):
        assert reg._student_t_sf(t, df) == pytest.approx(expected, abs=0.001)

    @pytest.mark.parametrize(
        ("f", "d1", "d2", "expected"),
        [(4.35, 1, 20, 0.05), (3.49, 2, 20, 0.05), (1.0, 5, 5, 0.5)],
    )
    def test_f_distribution_matches_tables(self, f, d1, d2, expected):
        assert reg._f_distribution_sf(f, d1, d2) == pytest.approx(expected, abs=0.001)

    def test_perfect_fit_recovers_exact_coefficients(self):
        x = np.arange(20, dtype=float)
        result = reg.fit_ols(x, 3.0 + 2.0 * x, ["x"])
        assert result.r_squared == pytest.approx(1.0)
        assert result.coefficients[0].value == pytest.approx(3.0)
        assert result.coefficients[1].value == pytest.approx(2.0)

    def test_known_slope_is_recovered_from_noisy_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 300)
        y = 1.5 + 0.8 * x + rng.normal(0, 1, 300)
        result = reg.fit_ols(x, y, ["x"])
        assert result.coefficients[1].value == pytest.approx(0.8, abs=0.15)
        assert result.coefficients[1].is_significant

    def test_pure_noise_is_not_significant(self):
        """The most important property: no signal must not look like signal."""
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = reg.fit_ols(x, y, ["x"])
        assert not result.coefficients[1].is_significant

    def test_r_squared_is_bounded(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = reg.fit_ols(x, y, ["x"])
        assert 0.0 <= result.r_squared <= 1.0

    def test_rows_with_nan_are_dropped(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        result = reg.fit_ols(x, y, ["x"])
        assert result.n_observations == 5

    def test_underpowered_fit_is_flagged(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.1, 5.9, 8.2])
        assert reg.fit_ols(x, y, ["x"]).is_underpowered

    def test_summary_always_carries_the_caveat(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 200)
        result = reg.fit_ols(x, 2 * x + rng.normal(0, 1, 200), ["x"])
        assert "not evidence" in result.summary()

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="Not enough observations"):
            reg.fit_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0]), ["x"])

    def test_multiple_predictors(self):
        rng = np.random.default_rng(6)
        x1 = rng.normal(0, 1, 200)
        x2 = rng.normal(0, 1, 200)
        y = 1.0 + 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, 200)
        result = reg.fit_ols(np.column_stack([x1, x2]), y, ["x1", "x2"])
        assert result.coefficients[1].value == pytest.approx(2.0, abs=0.2)
        assert result.coefficients[2].value == pytest.approx(-1.0, abs=0.2)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="rows"):
            reg.fit_ols(np.arange(10.0), np.arange(5.0), ["x"])


class TestCorrelation:
    def test_perfect_positive(self):
        x = np.arange(20, dtype=float)
        assert reg.correlate(x, 2 * x).correlation == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = np.arange(20, dtype=float)
        assert reg.correlate(x, -x).correlation == pytest.approx(-1.0)

    def test_constant_series_yields_nan(self):
        """Zero variance has no correlation, and must not raise."""
        x = np.ones(10)
        result = reg.correlate(x, np.arange(10, dtype=float))
        assert np.isnan(result.correlation)

    def test_too_few_points_yields_nan(self):
        result = reg.correlate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert np.isnan(result.correlation)
