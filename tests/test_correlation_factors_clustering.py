"""Tests for factor-model correlations, hierarchical clustering, and the PSD
guarantee of every structured estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlations import clustering, factors
from correlations import estimators as est
from correlations.stats import check_psd


def _sector_universe(t=1500, seed=0):
    """Market + two sectors of three assets each, known loadings."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, t)
    tech, energy = rng.normal(0, 0.008, t), rng.normal(0, 0.008, t)
    data = {}
    for i in range(3):
        data[f"T{i}"] = market + tech + rng.normal(0, 0.006, t)
        data[f"E{i}"] = market + energy + rng.normal(0, 0.006, t)
    index = pd.bdate_range("2015-01-01", periods=t)
    frame = pd.DataFrame(data, index=index)
    groups = {f"T{i}": "tech" for i in range(3)} | {f"E{i}": "energy" for i in range(3)}
    return frame, groups, pd.Series(market, index=index)


class TestPcaFactorModel:
    def test_recovers_the_factor_count(self):
        frame, _, _ = _sector_universe()
        # Market + 2 sectors on six assets has rank 2: the market loads equally
        # on both blocks, so it lies in the span of the two sector directions.
        result = factors.pca_factor_model(frame.to_numpy(), variance_target=0.85)
        assert result.n_factors == 2
        assert result.explained >= 0.85

    def test_one_factor_explains_an_equicorrelated_market(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2000, 1)) + 0.5 * rng.normal(size=(2000, 6))
        assert factors.pca_factor_model(x, 0.75).n_factors == 1

    def test_unit_diagonal_and_psd(self):
        frame, _, _ = _sector_universe()
        result = factors.pca_factor_model(frame.to_numpy())
        check = check_psd(result.correlation)
        assert check.is_psd and check.unit_diagonal


class TestMarketGroupFactorModel:
    def test_reproduces_within_and_across_sector_structure(self):
        """Regression: a time-series group factor biased the within-group
        correlation (true ~0.82) to 0.88 with the asset included and 0.72 with
        it left out. The covariance-form estimate must recover it."""
        frame, groups, market = _sector_universe()
        result = factors.market_group_factor_model(frame, groups, market)
        sample = frame.corr().to_numpy()
        assert result.factor_names == ("market", "energy", "tech")
        # The model is correctly specified here, so it should match the sample closely.
        assert np.abs(result.correlation - sample).max() < 0.02
        assert check_psd(result.correlation).is_psd

    def test_singleton_group_is_folded(self):
        frame, groups, market = _sector_universe()
        groups = dict(groups, T0="lonely")
        result = factors.market_group_factor_model(frame, groups, market)
        assert any("lonely" in w for w in result.warnings)
        assert "lonely" not in result.factor_names


class TestClustering:
    def test_hand_computed_three_points(self):
        # Distances: A-B 0.1, A-C 0.5, B-C 0.6. First merge A+B at 0.1, then
        # average linkage to C: (0.5 + 0.6) / 2 = 0.55.
        distance = np.array([[0.0, 0.1, 0.5], [0.1, 0.0, 0.6], [0.5, 0.6, 0.0]])
        linkage = clustering.average_linkage(distance)
        assert linkage[0].tolist() == pytest.approx([0, 1, 0.1, 2])
        assert linkage[1].tolist() == pytest.approx([2, 3, 0.55, 3])

    def test_blocks_end_up_adjacent(self):
        frame, _, _ = _sector_universe()
        result = clustering.cluster(frame.corr().to_numpy(), list(frame.columns))
        order = result.ordered_labels
        positions = {name: order.index(name) for name in order}
        tech = sorted(positions[f"T{i}"] for i in range(3))
        assert tech[-1] - tech[0] == 2  # the three tech names are contiguous

    def test_distance_is_a_metric_scale(self):
        d = clustering.correlation_distance(np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert d[0, 1] == pytest.approx(np.sqrt(0.5))
        d = clustering.correlation_distance(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        assert d[0, 1] == pytest.approx(1.0)


class TestStructuredEstimators:
    @pytest.mark.parametrize("method", ["ledoit_wolf", "denoised", "pca_factor"])
    def test_valid_matrix_on_random_data(self, method):
        rng = np.random.default_rng(7)
        values = 0.6 * rng.normal(size=(400, 1)) + rng.normal(size=(400, 25))
        frame = pd.DataFrame(values * 0.01, index=pd.bdate_range("2020-01-01", periods=400))
        estimate = est.estimate_correlation(frame, method=method)
        assert estimate.is_psd, estimate.warnings
        assert np.allclose(np.diag(estimate.matrix), 1.0)

    def test_group_factor_through_the_dispatcher(self):
        frame, groups, market = _sector_universe()
        estimate = est.estimate_correlation(frame, "group_factor", groups=groups, market=market)
        assert estimate.is_psd
        assert "market" in estimate.parameters["factors"]

    def test_identity_target_shrinks_toward_zero(self):
        frame, _, _ = _sector_universe(t=80)
        sample = est.pearson(frame).matrix.to_numpy()
        shrunk = est.ledoit_wolf_estimate(frame, target="identity").matrix.to_numpy()
        off = ~np.eye(6, dtype=bool)
        assert np.abs(shrunk[off]).mean() < np.abs(sample[off]).mean()

    def test_short_sample_is_nan_with_reason(self):
        frame, _, _ = _sector_universe(t=40)
        for method in ("ledoit_wolf", "denoised", "pca_factor"):
            estimate = est.estimate_correlation(frame, method=method)
            assert estimate.matrix.isna().all().all()
            assert any("60" in w for w in estimate.warnings)
