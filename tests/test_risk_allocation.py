"""Tests for the risk gate and position sizing.

Built on synthetic return series with known properties, so a Sharpe or a
correlation can be checked against what the construction guarantees.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from agents import allocation_decision as alloc
from agents import risk_manager as risk


def _series(mean: float, std: float, n: int = 400, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=index)


def _book(series: pd.Series, columns: int = 2, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            f"H{i}": rng.normal(0.0004, 0.011, len(series))
            for i in range(columns)
        },
        index=series.index,
    )


class TestCorrelation:
    def test_identical_series_correlate_at_one(self):
        candidate = _series(0.0008, 0.012)
        book = pd.DataFrame({"A": candidate})
        assert risk.compute_correlation(candidate, book) == pytest.approx(1.0)

    def test_inverted_series_correlate_at_minus_one(self):
        candidate = _series(0.0008, 0.012)
        book = pd.DataFrame({"A": -candidate})
        assert risk.compute_correlation(candidate, book) == pytest.approx(-1.0)

    def test_independent_series_correlate_near_zero(self):
        candidate = _series(0.0008, 0.012, seed=1)
        book = _book(candidate, seed=2)
        assert abs(risk.compute_correlation(candidate, book)) < 0.2

    def test_empty_portfolio_yields_nan(self):
        """No book is not zero correlation; it is an unanswerable question."""
        candidate = _series(0.0008, 0.012)
        assert np.isnan(risk.compute_correlation(candidate, pd.DataFrame()))

    def test_constant_series_yields_nan(self):
        candidate = _series(0.0008, 0.012)
        book = pd.DataFrame({"A": np.zeros(len(candidate))}, index=candidate.index)
        assert np.isnan(risk.compute_correlation(candidate, book))

    def test_non_overlapping_index_yields_nan(self):
        candidate = _series(0.0008, 0.012)
        other = pd.DataFrame(
            {"A": np.zeros(50)},
            index=pd.date_range("2030-01-01", periods=50, freq="B"),
        )
        assert np.isnan(risk.compute_correlation(candidate, other))


class TestRiskMetrics:
    def test_positive_drift_yields_positive_sharpe(self):
        metrics = risk.compute_risk_adjusted_metrics(_series(0.0012, 0.010))
        assert metrics["sharpe"] > 0

    def test_negative_drift_yields_negative_sharpe(self):
        metrics = risk.compute_risk_adjusted_metrics(_series(-0.0012, 0.010))
        assert metrics["sharpe"] < 0

    def test_sortino_exceeds_sharpe_when_downside_is_muted(self):
        """Downside deviation is below total deviation for a right-skewed series."""
        rng = np.random.default_rng(5)
        # Small frequent losses, occasional large gains.
        returns = pd.Series(
            np.where(rng.random(500) < 0.8, -0.002, 0.012) + rng.normal(0, 0.001, 500)
        )
        metrics = risk.compute_risk_adjusted_metrics(returns)
        assert metrics["sortino"] > metrics["sharpe"]

    def test_zero_volatility_does_not_divide_by_zero(self):
        metrics = risk.compute_risk_adjusted_metrics(pd.Series([0.001] * 100))
        assert metrics["sharpe"] == 0.0
        assert metrics["sortino"] == 0.0

    def test_drawdown_is_negative_after_a_decline(self):
        returns = pd.Series([0.02] * 20 + [-0.03] * 20)
        assert risk.compute_risk_adjusted_metrics(returns)["max_drawdown_pct"] < 0

    def test_empty_series_is_handled(self):
        metrics = risk.compute_risk_adjusted_metrics(pd.Series([], dtype=float))
        assert metrics["observations"] == 0
        assert metrics["sharpe"] == 0.0


class TestRiskGate:
    def test_strong_candidate_is_approved(self):
        candidate = _series(0.0012, 0.010, seed=3)
        assessment = risk.assess("GOOD", candidate, _book(candidate))
        assert assessment.approved
        assert assessment.sharpe_ratio > 0

    def test_negative_sharpe_is_rejected(self):
        candidate = _series(-0.0015, 0.015, seed=4)
        assessment = risk.assess("BAD", candidate, _book(candidate))
        assert not assessment.approved
        assert "Sharpe" in assessment.reason

    def test_duplicate_of_the_book_is_rejected(self):
        """A position that moves with the book concentrates rather than diversifies."""
        candidate = _series(0.0012, 0.010, seed=3)
        book = pd.DataFrame({"A": candidate * 0.99}, index=candidate.index)
        assessment = risk.assess("DUPE", candidate, book)
        assert not assessment.approved
        assert "orrelation" in assessment.reason

    def test_inverse_correlation_is_also_rejected(self):
        """A perfect hedge is as much a duplicate as a perfect copy."""
        candidate = _series(0.0012, 0.010, seed=3)
        book = pd.DataFrame({"A": -candidate}, index=candidate.index)
        assessment = risk.assess("INVERSE", candidate, book)
        assert not assessment.approved

    def test_missing_portfolio_skips_the_correlation_check(self):
        candidate = _series(0.0012, 0.010, seed=3)
        assessment = risk.assess("FIRST", candidate, None)
        assert assessment.approved
        assert any("skipped" in w for w in assessment.warnings)

    def test_short_history_raises_a_warning_not_a_rejection(self):
        candidate = _series(0.0012, 0.010, n=30, seed=3)
        assessment = risk.assess("SHORT", candidate, None)
        assert any("observations" in w for w in assessment.warnings)

    def test_rejection_always_states_a_reason(self):
        candidate = _series(-0.002, 0.02, seed=6)
        assessment = risk.assess("X", candidate, None)
        assert not assessment.approved
        assert len(assessment.reason) > 20

    def test_summary_renders(self):
        assessment = risk.assess("X", _series(0.001, 0.01), None)
        assert "Risk assessment" in assessment.summary()


class TestSizing:
    @pytest.fixture
    def approved(self) -> dict:
        return {
            "approved": True,
            "annualized_volatility_pct": 20.0,
            "annualized_return_pct": 15.0,
            "correlation_to_portfolio": 0.0,
        }

    def test_rejected_candidate_is_sized_at_zero(self, approved):
        rejected = {**approved, "approved": False}
        decision = alloc.size_position("X", 1.0, rejected, 100_000)
        assert decision.position_size_pct == 0.0
        assert "rejected" in decision.rationale.lower()

    def test_volatility_targeting_shrinks_with_volatility(self, approved):
        """The whole point: a more volatile name gets less capital."""
        # Both must land below the 20% cap, or the cap hides the difference.
        calm = alloc.size_position(
            "CALM", 1.0, {**approved, "annualized_volatility_pct": 90.0}, 100_000
        )
        wild = alloc.size_position(
            "WILD", 1.0, {**approved, "annualized_volatility_pct": 200.0}, 100_000
        )
        assert calm.position_size_pct > wild.position_size_pct

    def test_volatility_targeting_equalises_risk_contribution(self):
        """Two names at different volatilities should contribute similar risk."""
        base = {"approved": True, "annualized_return_pct": 10.0,
                "correlation_to_portfolio": 0.0}
        a = alloc.size_position(
            "A", 0.5, {**base, "annualized_volatility_pct": 100.0}, 100_000
        )
        b = alloc.size_position(
            "B", 0.5, {**base, "annualized_volatility_pct": 200.0}, 100_000
        )
        assert a.expected_volatility_contribution_pct == pytest.approx(
            b.expected_volatility_contribution_pct, rel=0.05
        )

    def test_conviction_scales_the_position(self, approved):
        low = alloc.size_position("X", 0.2, approved, 100_000)
        high = alloc.size_position("X", 0.9, approved, 100_000)
        assert high.position_size_pct > low.position_size_pct

    def test_position_never_exceeds_the_cap(self, approved):
        tiny_vol = {**approved, "annualized_volatility_pct": 2.0}
        decision = alloc.size_position("X", 1.0, tiny_vol, 100_000)
        assert decision.position_size_pct <= alloc.MAX_POSITION_PCT
        assert any("Capped" in c for c in decision.constraints_applied)

    def test_unconstrained_size_is_reported_alongside(self, approved):
        """The cap should be visible, not silent."""
        tiny_vol = {**approved, "annualized_volatility_pct": 2.0}
        decision = alloc.size_position("X", 1.0, tiny_vol, 100_000)
        assert decision.unconstrained_size_pct > decision.position_size_pct

    def test_correlation_reduces_the_size(self, approved):
        # High volatility keeps both below the cap, which would otherwise mask
        # the haircut.
        volatile = {**approved, "annualized_volatility_pct": 120.0}
        independent = alloc.size_position("X", 0.5, volatile, 100_000)
        correlated = alloc.size_position(
            "X", 0.5, {**volatile, "correlation_to_portfolio": 0.9}, 100_000
        )
        assert correlated.position_size_pct < independent.position_size_pct

    def test_kelly_is_fractional_not_full(self):
        """Full Kelly on estimated inputs overbets; the implementation halves it."""
        edge = {
            "approved": True,
            "annualized_volatility_pct": 30.0,
            "annualized_return_pct": 20.0,
            "correlation_to_portfolio": 0.0,
        }
        decision = alloc.size_position(
            "X", 1.0, edge, 100_000, method=alloc.SizingMethod.KELLY
        )
        full_kelly = (0.20 / 0.30**2) * 100.0
        assert decision.unconstrained_size_pct == pytest.approx(
            full_kelly * alloc.KELLY_FRACTION, rel=0.01
        )

    def test_kelly_declines_a_negative_edge(self):
        no_edge = {
            "approved": True,
            "annualized_volatility_pct": 30.0,
            "annualized_return_pct": -5.0,
            "correlation_to_portfolio": 0.0,
        }
        decision = alloc.size_position(
            "X", 1.0, no_edge, 100_000, method=alloc.SizingMethod.KELLY
        )
        assert decision.position_size_pct == 0.0

    def test_tiny_position_is_dropped(self, approved):
        decision = alloc.size_position("X", 0.001, approved, 100_000)
        assert decision.position_size_pct == 0.0
        assert not decision.is_actionable

    def test_costs_are_charged_on_both_sides(self, approved):
        decision = alloc.size_position(
            "X", 0.5, approved, 100_000, estimated_cost_bps=10.0
        )
        expected = decision.position_value * 0.001 * 2
        assert decision.estimated_transaction_cost == pytest.approx(expected, rel=0.01)

    def test_zero_volatility_does_not_divide_by_zero(self, approved):
        decision = alloc.size_position(
            "X", 1.0, {**approved, "annualized_volatility_pct": 0.0}, 100_000
        )
        assert decision.position_size_pct == 0.0

    def test_negative_equity_is_rejected(self, approved):
        with pytest.raises(ValueError, match="positive"):
            alloc.size_position("X", 0.5, approved, -100.0)

    def test_risk_assessment_dataclass_can_be_passed_through(self):
        """The two modules must interoperate without manual translation."""
        candidate = _series(0.0012, 0.010, seed=3)
        assessment = risk.assess("GOOD", candidate, None)
        decision = alloc.size_position("GOOD", 0.7, asdict(assessment), 100_000)
        assert decision.position_size_pct > 0

    def test_summary_renders(self, approved):
        decision = alloc.size_position("X", 0.5, approved, 100_000)
        assert "Allocation" in decision.summary()
