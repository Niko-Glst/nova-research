"""Tests for the regime-split outcome distribution.

Built from synthetic series with known properties, so a probability can be
checked against what the construction guarantees rather than against another
run of the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests import outcome_distribution as od


def _frame(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    prices = start * np.cumprod(1.0 + returns)
    index = pd.date_range("2022-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=index)


def _two_regime_returns(seed: int = 7) -> np.ndarray:
    """500 calm days followed by 300 turbulent ones."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0006, 0.008, 500)
    turbulent = rng.normal(-0.0004, 0.030, 300)
    return np.concatenate([calm, turbulent])


class TestRegimeClassification:
    def test_calm_and_volatile_days_are_separated(self):
        returns = pd.Series(_two_regime_returns())
        labels, threshold = od.classify_regimes(returns)
        assert set(labels.unique()) <= {"calm", "volatile", "unknown"}
        assert threshold > 0

    def test_turbulent_period_is_mostly_labelled_volatile(self):
        returns = pd.Series(_two_regime_returns())
        labels, _ = od.classify_regimes(returns)
        # The last 300 days are the turbulent block.
        tail = labels.iloc[-250:]
        assert (tail == "volatile").mean() > 0.7

    def test_split_percentile_controls_the_balance(self):
        returns = pd.Series(_two_regime_returns())
        labels_low, _ = od.classify_regimes(returns, split_percentile=30)
        labels_high, _ = od.classify_regimes(returns, split_percentile=90)
        assert (labels_low == "volatile").sum() > (labels_high == "volatile").sum()

    def test_short_series_yields_unknown(self):
        labels, threshold = od.classify_regimes(pd.Series([0.01, 0.02]))
        assert (labels == "unknown").all()
        assert threshold == 0.0


class TestBuildDistribution:
    @pytest.fixture
    def distribution(self):
        return od.build_distribution(
            "TEST", _frame(_two_regime_returns()), num_simulations=1500, seed=1
        )

    def test_all_three_regimes_are_reported(self, distribution):
        assert set(distribution.regimes) == {"calm", "volatile", "all"}

    def test_volatile_regime_has_higher_volatility(self, distribution):
        calm = distribution.regimes["calm"].annualized_volatility_pct
        volatile = distribution.regimes["volatile"].annualized_volatility_pct
        assert volatile > calm

    def test_volatile_regime_has_wider_distribution(self, distribution):
        """The whole point of splitting: the tails differ, not just the middle."""
        calm = distribution.regimes["calm"]
        volatile = distribution.regimes["volatile"]
        calm_spread = calm.return_percentiles[95] - calm.return_percentiles[5]
        volatile_spread = volatile.return_percentiles[95] - volatile.return_percentiles[5]
        assert volatile_spread > calm_spread

    def test_volatile_regime_has_higher_drawdown_risk(self, distribution):
        calm = distribution.regimes["calm"].probability_of_drawdown_beyond[30.0]
        volatile = distribution.regimes["volatile"].probability_of_drawdown_beyond[30.0]
        assert volatile > calm

    def test_percentiles_are_ordered(self, distribution):
        for regime in distribution.regimes.values():
            values = [regime.return_percentiles[p] for p in od.DEFAULT_PERCENTILES]
            assert values == sorted(values)

    def test_return_probabilities_decrease_with_threshold(self, distribution):
        """P(return > x) must fall as x rises; anything else is a bug."""
        for regime in distribution.regimes.values():
            probabilities = [
                regime.probability_of_return_above[t] for t in od.RETURN_THRESHOLDS
            ]
            assert probabilities == sorted(probabilities, reverse=True)

    def test_drawdown_probabilities_decrease_with_threshold(self, distribution):
        for regime in distribution.regimes.values():
            probabilities = [
                regime.probability_of_drawdown_beyond[t] for t in od.DRAWDOWN_THRESHOLDS
            ]
            assert probabilities == sorted(probabilities, reverse=True)

    def test_probabilities_are_within_zero_and_one(self, distribution):
        for regime in distribution.regimes.values():
            for value in regime.probability_of_return_above.values():
                assert 0.0 <= value <= 1.0
            for value in regime.probability_of_drawdown_beyond.values():
                assert 0.0 <= value <= 1.0

    def test_loss_probability_complements_the_zero_threshold(self, distribution):
        for regime in distribution.regimes.values():
            expected = 1.0 - regime.probability_of_return_above[0.0]
            assert regime.probability_of_loss == pytest.approx(expected)

    def test_expected_shortfall_is_below_the_fifth_percentile(self, distribution):
        """The mean of the worst tail cannot exceed the tail's boundary."""
        for regime in distribution.regimes.values():
            assert regime.expected_shortfall_pct <= regime.return_percentiles[5] + 1e-6

    def test_same_seed_gives_identical_probabilities(self):
        frame = _frame(_two_regime_returns())
        first = od.build_distribution("TEST", frame, num_simulations=800, seed=3)
        second = od.build_distribution("TEST", frame, num_simulations=800, seed=3)
        assert (
            first.regimes["all"].return_percentiles
            == second.regimes["all"].return_percentiles
        )

    def test_threshold_is_reported_so_the_label_can_be_read(self, distribution):
        """'Calm' means nothing without the number it was split on."""
        assert distribution.regime_threshold_pct > 0
        assert f"{distribution.regime_threshold_pct:.0f}%" in distribution.summary()

    def test_caveats_always_mention_the_core_assumption(self, distribution):
        assert any("resembles the past" in c for c in distribution.caveats)

    def test_summary_renders(self, distribution):
        text = distribution.summary()
        assert "OUTCOME PROBABILITIES" in text
        assert "PROBABILITY OF DRAWDOWN BEYOND" in text


class TestDrift:
    def test_positive_drift_yields_a_positive_median(self):
        rng = np.random.default_rng(11)
        returns = rng.normal(0.0008, 0.012, 600)
        distribution = od.build_distribution(
            "UP", _frame(returns), num_simulations=1500, seed=2
        )
        assert distribution.regimes["all"].median_return_pct > 0

    def test_negative_drift_yields_a_negative_median(self):
        rng = np.random.default_rng(12)
        returns = rng.normal(-0.0008, 0.012, 600)
        distribution = od.build_distribution(
            "DOWN", _frame(returns), num_simulations=1500, seed=2
        )
        assert distribution.regimes["all"].median_return_pct < 0


class TestValidation:
    def test_empty_history_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            od.build_distribution("X", pd.DataFrame({"Close": []}))

    def test_missing_close_column_is_rejected(self):
        with pytest.raises(ValueError, match="Close"):
            od.build_distribution("X", pd.DataFrame({"Open": [1.0, 2.0, 3.0]}))

    def test_too_little_history_is_rejected(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="at least"):
            od.build_distribution("X", _frame(rng.normal(0, 0.01, 20)))

    def test_long_horizon_relative_to_history_is_flagged(self):
        rng = np.random.default_rng(2)
        distribution = od.build_distribution(
            "X", _frame(rng.normal(0, 0.01, 200)), horizon_days=252,
            num_simulations=500, seed=1,
        )
        assert any("long relative to" in c for c in distribution.caveats)


class TestHistoricalDrawdowns:
    def test_worst_drawdown_is_negative_after_a_fall(self):
        prices = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 120, 60)])
        frame = pd.DataFrame(
            {"Close": prices}, index=pd.date_range("2023-01-01", periods=len(prices))
        )
        result = od.historical_drawdowns(frame)
        assert result["worst_drawdown_pct"] < -35

    def test_monotonic_rise_has_no_drawdown(self):
        frame = _frame(np.full(200, 0.001))
        result = od.historical_drawdowns(frame)
        assert result["worst_drawdown_pct"] == pytest.approx(0.0)

    def test_reports_time_spent_beyond_each_threshold(self):
        prices = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 100, 100)])
        frame = pd.DataFrame(
            {"Close": prices}, index=pd.date_range("2023-01-01", periods=len(prices))
        )
        result = od.historical_drawdowns(frame)
        assert 0.0 <= result["days_beyond_20pct"] <= 1.0
