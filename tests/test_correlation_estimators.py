"""Tests for correlations.estimators on synthetic data with known properties."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.risk_manager import MIN_OBSERVATIONS
from correlations import estimators as est


def _frame(values: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    names = names or [f"A{i}" for i in range(values.shape[1])]
    index = pd.bdate_range("2020-01-01", periods=values.shape[0])
    return pd.DataFrame(values, index=index, columns=names)


def _correlated_pair(rho: float, n: int, seed: int = 0) -> pd.DataFrame:
    """Two series with population correlation rho, via the Cholesky factor."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
    return _frame(np.column_stack([x, y]) * 0.01, ["X", "Y"])


class TestPublishedExample:
    def test_investopedia_worked_example(self):
        """Pearson on the published worked example: r = 3,913 / 7,248.4 = 0.54.

        X and Y from Investopedia's "Correlation" article. They are levels in
        the article, used here only as two numeric series; MIN_OBSERVATIONS is
        bypassed by repeating the seven points, which leaves r unchanged.
        """
        x = [41, 19, 23, 40, 55, 57, 33]
        y = [94, 60, 74, 71, 82, 76, 61]
        repeats = math.ceil(MIN_OBSERVATIONS / 7)
        frame = _frame(np.column_stack([x * repeats, y * repeats]).astype(float), ["X", "Y"])
        estimate = est.pearson(frame)
        assert estimate.pair("X", "Y") == pytest.approx(3913 / math.sqrt(8914 * 5894))
        assert round(estimate.pair("X", "Y"), 2) == 0.54


class TestSampleEstimators:
    def test_recovers_known_correlation(self):
        frame = _correlated_pair(0.6, 5000)
        estimate = est.pearson(frame)
        # Standard error of r near rho = 0.6 with n = 5000 is (1 - rho^2) / sqrt(n) ~ 0.009.
        assert estimate.pair("X", "Y") == pytest.approx(0.6, abs=0.04)
        assert estimate.is_psd
        assert estimate.covariance is not None

    def test_rank_methods_on_a_normal_pair(self):
        # On a bivariate normal: Spearman = (6/pi) asin(rho/2), Kendall = (2/pi) asin(rho).
        rho = 0.6
        frame = _correlated_pair(rho, 5000, seed=3)
        assert est.spearman(frame).pair("X", "Y") == pytest.approx(
            6 / math.pi * math.asin(rho / 2), abs=0.04
        )
        assert est.kendall(frame).pair("X", "Y") == pytest.approx(
            2 / math.pi * math.asin(rho), abs=0.04
        )

    def test_average_ranks_hand_computed(self):
        # x = 1,2,2,3 ranks to 1, 2.5, 2.5, 4 and y = 1,3,2,3 to 1, 3.5, 2, 3.5.
        # Deviations from 2.5: (-1.5, 0, 0, 1.5) and (-1.5, 1, -0.5, 1);
        # r = 3.75 / sqrt(4.5 * 4.5) = 0.8333.
        base = pd.DataFrame([[1, 1], [2, 3], [2, 2], [3, 3]], columns=["X", "Y"], dtype=float)
        ranks = base.rank(method="average")
        assert ranks["X"].tolist() == [1, 2.5, 2.5, 4]
        assert ranks.corr().loc["X", "Y"] == pytest.approx(3.75 / 4.5)

    def test_spearman_is_pearson_on_average_ranks_with_heavy_ties(self):
        rng = np.random.default_rng(9)
        values = rng.integers(-3, 4, size=(300, 2)).astype(float)  # 7 levels: many ties
        frame = _frame(values, ["X", "Y"])
        expected = np.corrcoef(frame.rank(method="average").to_numpy(), rowvar=False)[0, 1]
        assert est.spearman(frame).pair("X", "Y") == pytest.approx(expected)

    def test_pairs_match_the_matrix_and_carry_bh(self):
        frame = _correlated_pair(0.3, 500)
        estimate = est.pearson(frame)
        (pair,) = estimate.pairs
        assert pair.correlation == pytest.approx(estimate.pair("X", "Y"))
        assert pair.n_observations == 500
        assert pair.significant and pair.significant_adjusted
        assert pair.p_adjusted == pytest.approx(pair.p_value)  # one test: no correction

    def test_independent_series_are_not_significant(self):
        frame = _correlated_pair(0.0, 400, seed=11)
        (pair,) = est.pearson(frame).pairs
        assert not pair.significant_adjusted


class TestGuards:
    def test_below_minimum_observations_is_nan_with_reason(self):
        frame = _correlated_pair(0.5, MIN_OBSERVATIONS - 1)
        for method in ("pearson", "spearman", "kendall", "ewma"):
            estimate = est.estimate_correlation(frame, method=method)
            assert estimate.matrix.isna().all().all()
            assert any(str(MIN_OBSERVATIONS) in w for w in estimate.warnings)
            assert not estimate.is_psd

    def test_constant_column_is_nan_not_zero(self):
        frame = _correlated_pair(0.5, 300)
        frame["FLAT"] = 0.0
        estimate = est.pearson(frame)
        assert math.isnan(estimate.pair("X", "FLAT"))
        assert any("zero variance" in w for w in estimate.warnings)
        assert estimate.pair("X", "Y") == pytest.approx(estimate.pair("Y", "X"))

    def test_missing_rows_are_dropped_and_noted(self):
        frame = _correlated_pair(0.5, 300)
        frame.iloc[5, 0] = np.nan
        estimate = est.pearson(frame)
        assert estimate.n_observations == 299
        assert any("dropped" in w for w in estimate.warnings)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            est.estimate_correlation(_correlated_pair(0.5, 100), method="magic")


class TestPsdForEveryEstimator:
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", "ewma", "rolling"])
    def test_random_universe_gives_valid_matrix(self, method):
        rng = np.random.default_rng(7)
        factor = rng.normal(size=(400, 1))
        values = 0.6 * factor + rng.normal(size=(400, 6))
        estimate = est.estimate_correlation(_frame(values * 0.01), method=method)
        assert estimate.is_psd, estimate.warnings
        assert np.allclose(np.diag(estimate.matrix.to_numpy()), 1.0)
        assert np.allclose(estimate.matrix, estimate.matrix.T)


class TestEwma:
    def test_weights_and_effective_sample(self):
        weights = est.ewma_weights(10_000, 0.94)
        assert weights.sum() == pytest.approx(1.0)
        assert weights[-1] > weights[0]
        # Kish ESS of an infinite EWMA is (1 + lam) / (1 - lam) = 32.3 for 0.94.
        assert est.effective_sample_size(weights) == pytest.approx(1.94 / 0.06, rel=1e-3)

    def test_small_effective_sample_is_warned(self):
        estimate = est.ewma(_correlated_pair(0.5, 500), frequency="daily")
        assert estimate.parameters["lambda"] == 0.94
        assert any("Effective sample size" in w for w in estimate.warnings)

    def test_tracks_a_regime_switch(self):
        first = _correlated_pair(0.8, 1000, seed=1)
        second = _correlated_pair(-0.4, 300, seed=2)
        second.index = pd.bdate_range(first.index[-1] + pd.Timedelta(days=1), periods=300)
        frame = pd.concat([first, second])
        assert est.ewma(frame).pair("X", "Y") < 0.0
        assert est.pearson(frame).pair("X", "Y") > 0.4


class TestRolling:
    def test_regime_switch_shows_in_the_path(self):
        first = _correlated_pair(0.8, 600, seed=4)
        second = _correlated_pair(-0.5, 600, seed=5)
        second.index = pd.bdate_range(first.index[-1] + pd.Timedelta(days=1), periods=600)
        rolling = est.rolling_paths(pd.concat([first, second]), window=120)
        (path,) = rolling.paths
        assert path.maximum > 0.6 and path.minimum < -0.3
        assert path.latest < 0.0
        assert rolling.latest is not None and rolling.latest.is_psd

    def test_window_below_minimum_is_refused(self):
        rolling = est.rolling_paths(_correlated_pair(0.5, 500), window=30)
        assert not rolling.paths
        assert any("minimum" in w for w in rolling.warnings)
