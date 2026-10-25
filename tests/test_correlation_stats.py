"""Tests for correlations.stats: Kendall tau-b, Benjamini-Hochberg, PSD check.

Expected values are worked by hand in the comments, not taken from a second
implementation, so a shared mistake cannot pass as agreement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from correlations import stats


class TestKendall:
    def test_hand_computed_no_ties(self):
        # x = 1..5, y = 3,1,2,5,4. Of the 10 pairs, 7 are concordant and 3
        # discordant ((1,2), (1,3), (4,5)), so tau = (7 - 3) / 10 = 0.4.
        result = stats.kendall_tau_b([1, 2, 3, 4, 5], [3, 1, 2, 5, 4])
        assert result.tau == pytest.approx(0.4)

    def test_hand_computed_p_value(self):
        # S = 4, Var(S) = n(n-1)(2n+5)/18 = 5*4*15/18 = 16.667, z = 0.9798,
        # two-sided p = erfc(z / sqrt 2) = 0.3272.
        result = stats.kendall_tau_b([1, 2, 3, 4, 5], [3, 1, 2, 5, 4])
        assert result.p_value == pytest.approx(0.3272, abs=1e-4)

    def test_hand_computed_with_ties(self):
        # x = 1,2,2,3 and y = 1,3,2,3. Concordant: (1,2) (1,3) (1,4) (3,4);
        # (2,3) ties in x and (2,4) ties in y, so S = 4. n0 = 6, one tied pair
        # in each variable: tau-b = 4 / sqrt(5 * 5) = 0.8.
        result = stats.kendall_tau_b([1, 2, 2, 3], [1, 3, 2, 3])
        assert result.tau == pytest.approx(0.8)

    def test_perfect_agreement_and_reversal(self):
        x = np.arange(20.0)
        assert stats.kendall_tau_b(x, x).tau == pytest.approx(1.0)
        assert stats.kendall_tau_b(x, -x).tau == pytest.approx(-1.0)

    def test_chunking_does_not_change_the_answer(self, monkeypatch):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=300), rng.normal(size=300)
        whole = stats.kendall_tau_b(x, y).tau
        monkeypatch.setattr(stats, "KENDALL_CHUNK", 7)
        assert stats.kendall_tau_b(x, y).tau == pytest.approx(whole)

    def test_matrix_matches_pairwise(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(200, 3))
        values[:, 1] += values[:, 0]
        tau, p = stats.kendall_matrix(values)
        single = stats.kendall_tau_b(values[:, 0], values[:, 1])
        assert tau[0, 1] == pytest.approx(single.tau)
        assert p[0, 1] == pytest.approx(single.p_value)
        assert np.allclose(np.diag(tau), 1.0)

    def test_constant_series_is_undefined(self):
        assert math.isnan(stats.kendall_tau_b([1, 1, 1, 1], [1, 2, 3, 4]).tau)

    def test_built_in_matrix_matches_pairwise(self):
        rng = np.random.default_rng(4)
        values = np.round(rng.normal(size=(150, 3)), 1)  # rounding creates ties
        tau, p = stats.kendall_matrix(values, use_scipy=False)
        single = stats.kendall_tau_b(values[:, 0], values[:, 2])
        assert tau[0, 2] == pytest.approx(single.tau, abs=1e-14)
        assert p[0, 2] == pytest.approx(single.p_value, abs=1e-14)

    def test_scipy_and_built_in_agree_with_ties(self):
        pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(5)
        values = np.round(rng.standard_t(4, size=(400, 5)) * 3) / 3
        values[:, 1] += 0.5 * values[:, 0]
        fast_tau, fast_p = stats.kendall_matrix(values, use_scipy=True)
        slow_tau, slow_p = stats.kendall_matrix(values, use_scipy=False)
        assert np.abs(fast_tau - slow_tau).max() < 1e-12
        assert np.abs(fast_p - slow_p).max() < 1e-12

    def test_blocked_scipy_falls_back_and_says_how_to_fix(self, monkeypatch):
        from correlations.estimators import kendall
        import pandas as pd

        monkeypatch.setattr(stats, "_scipy_kendalltau",
                            lambda: (None, "ImportError: DLL load failed"))
        assert stats.kendall_backend() == "built-in"
        rng = np.random.default_rng(6)
        frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
        estimate = kendall(frame)
        assert estimate.parameters["backend"] == "built-in"
        assert any(stats.SCIPY_FIX in w for w in estimate.warnings)
        assert np.isfinite(estimate.matrix.to_numpy()).all()


class TestBenjaminiHochberg:
    def test_hand_computed(self):
        # Sorted: 0.005 (rank 1), 0.01 (2), 0.03 (3), 0.04 (4); m = 4.
        # p * m / rank: 0.02, 0.02, 0.04, 0.04; running minimum from the top
        # leaves them unchanged.
        adjusted, reject = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
        assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])
        assert reject.tolist() == [True, True, True, True]

    def test_running_minimum_from_the_top(self):
        # Ranked: 0.01*3/1 = 0.03, 0.04*3/2 = 0.06, 0.045*3/3 = 0.045.
        # The middle value takes the minimum of itself and those above: 0.045.
        adjusted, reject = stats.benjamini_hochberg([0.01, 0.04, 0.045])
        assert adjusted == pytest.approx([0.03, 0.045, 0.045])
        assert reject.tolist() == [True, True, True]

    def test_correction_removes_marginal_findings(self):
        adjusted, reject = stats.benjamini_hochberg([0.04] + [0.5] * 9)
        assert adjusted[0] == pytest.approx(0.4)
        assert not reject.any()

    def test_nan_is_not_a_test(self):
        adjusted, reject = stats.benjamini_hochberg([0.01, float("nan"), 0.02])
        assert math.isnan(adjusted[1]) and not reject[1]
        # m = 2, not 3: 0.01 * 2 / 1 = 0.02, 0.02 * 2 / 2 = 0.02.
        assert adjusted[0] == pytest.approx(0.02)

    def test_capped_at_one(self):
        adjusted, _ = stats.benjamini_hochberg([0.9, 0.95])
        assert (adjusted <= 1.0).all()


class TestPsd:
    def test_valid_correlation_matrix(self):
        check = stats.check_psd(np.array([[1.0, 0.5], [0.5, 1.0]]))
        assert check.is_psd and check.unit_diagonal and not check.warnings

    def test_impossible_matrix_is_flagged(self):
        # Three assets cannot all be correlated at -0.9 with each other.
        m = np.full((3, 3), -0.9)
        np.fill_diagonal(m, 1.0)
        check = stats.check_psd(m)
        assert not check.is_psd
        assert check.min_eigenvalue < 0
        assert "negative variance" in check.warnings[0]

    def test_nan_matrix(self):
        check = stats.check_psd(np.array([[1.0, np.nan], [np.nan, 1.0]]))
        assert not check.is_psd and "NaN" in check.warnings[0]
