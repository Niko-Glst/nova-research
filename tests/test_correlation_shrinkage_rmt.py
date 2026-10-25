"""Tests for Ledoit-Wolf shrinkage and random-matrix denoising.

Reference values come from independent code, not from this implementation:

- Identity target: scikit-learn 1.x `ledoit_wolf_shrinkage`. Its docstring
  publishes the first case (shrinkage 0.23 for RandomState(0) draws from
  [[.4, .2], [.2, .8]], 50 samples); the second was produced by running that
  function's source on the X1 data below.
- Constant-correlation target: PyPortfolioOpt 1.x
  `CovarianceShrinkage._ledoit_wolf_constant_correlation`, run on X1 with the
  sample covariance normalised by 1/T as in the Ledoit-Wolf (2004) paper.
  Unmodified, PyPortfolioOpt uses a 1/(T-1) covariance and gives 0.5182 instead
  of 0.5270; with the normalisation aligned the two agree to 16 digits.

Neither library is a project dependency: scipy, which both import, is blocked
on this machine, so the functions' source was executed with numpy alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlations import rmt, shrinkage
from correlations.stats import check_psd


def _x1() -> np.ndarray:
    rng = np.random.default_rng(123)
    factor = rng.normal(size=(120, 1))
    noise = rng.normal(size=(120, 5)) * np.array([1, 1.5, 0.7, 1.2, 0.9])
    return 0.6 * factor + noise


class TestLedoitWolfIdentity:
    def test_published_docstring_example(self):
        rng = np.random.RandomState(0)
        x = rng.multivariate_normal(mean=[0, 0], cov=[[0.4, 0.2], [0.2, 0.8]], size=50)
        result = shrinkage.shrink_identity(x)
        assert round(result.intensity, 2) == 0.23
        assert result.intensity == pytest.approx(0.23025406200239137, abs=1e-12)

    def test_matches_scikit_learn(self):
        assert shrinkage.shrink_identity(_x1()).intensity == pytest.approx(
            0.20984183358602412, abs=1e-12
        )

    def test_shrunk_matrix_is_the_convex_combination(self):
        x = _x1()
        result = shrinkage.shrink_identity(x)
        centered = x - x.mean(axis=0)
        sample = centered.T @ centered / len(x)
        mu = np.trace(sample) / sample.shape[0]
        expected = (1 - result.intensity) * sample + result.intensity * mu * np.eye(5)
        assert np.allclose(result.covariance, expected)


class TestLedoitWolfConstantCorrelation:
    def test_matches_pyportfolioopt(self):
        result = shrinkage.shrink_constant_correlation(_x1())
        assert result.intensity == pytest.approx(0.5269865214893996, abs=1e-12)
        assert result.covariance[0, 1] == pytest.approx(0.09292426152307623, abs=1e-12)
        assert result.covariance[2, 4] == pytest.approx(0.22087959477859065, abs=1e-12)

    def test_variances_are_untouched(self):
        x = _x1()
        result = shrinkage.shrink_constant_correlation(x)
        assert np.allclose(np.diag(result.covariance), x.var(axis=0))

    def test_correct_target_is_shrunk_to_fully(self):
        # Equicorrelated population: the constant-correlation target is the
        # truth, so the optimal estimate is the target itself (intensity 1).
        rng = np.random.default_rng(6)
        common = rng.normal(size=(5_000, 1))
        x = common + rng.normal(size=(5_000, 4))
        assert shrinkage.shrink_constant_correlation(x).intensity == pytest.approx(1.0)

    def test_wrong_target_is_trusted_less_as_data_grows(self):
        # Two blocks with different correlations: the single-average target is
        # misspecified, so more data should mean less shrinkage toward it.
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(20_000, 1)), rng.normal(size=(20_000, 1))
        x = np.hstack([a + rng.normal(size=(20_000, 3)), 0.3 * b + rng.normal(size=(20_000, 3))])
        small = shrinkage.shrink_constant_correlation(x[:60]).intensity
        large = shrinkage.shrink_constant_correlation(x).intensity
        assert 0.0 <= large < 0.05 < small <= 1.0

    def test_zero_variance_column_is_refused(self):
        x = _x1()
        x[:, 2] = 1.0
        with pytest.raises(ValueError, match="zero variance"):
            shrinkage.shrink_constant_correlation(x)

    def test_unknown_target(self):
        with pytest.raises(ValueError, match="Unknown shrinkage target"):
            shrinkage.ledoit_wolf(_x1(), target="banana")


class TestMarchenkoPastur:
    def test_density_integrates_to_one(self):
        lo, hi = rmt.marchenko_pastur_bounds(1.0, 4.0)
        x = np.linspace(lo, hi, 200_001)
        mass = np.trapezoid(rmt.marchenko_pastur_pdf(x, 1.0, 4.0), x)
        assert mass == pytest.approx(1.0, abs=1e-3)

    def test_bounds(self):
        # q = 4: (1 -+ 1/2)^2 = 0.25 and 2.25.
        assert rmt.marchenko_pastur_bounds(1.0, 4.0) == pytest.approx((0.25, 2.25))

    def test_pure_noise_eigenvalues_fall_inside_the_band(self):
        rng = np.random.default_rng(0)
        t, n = 1000, 50
        corr = np.corrcoef(rng.normal(size=(t, n)), rowvar=False)
        result = rmt.denoise(corr, t)
        assert result.sigma2 == pytest.approx(1.0, abs=0.15)
        lo, hi = rmt.marchenko_pastur_bounds(1.0, t / n)
        assert result.eigenvalues.max() < hi * 1.1
        assert result.eigenvalues.min() > lo * 0.9

    def test_pure_noise_has_no_signal_across_draws(self):
        """Regression: with the asymptotic lambda_plus as the edge, 2 of 20 noise
        draws produced a false 'signal' eigenvalue. The Tracy-Widom edge must
        not."""
        false_signals = 0
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            corr = np.corrcoef(rng.normal(size=(1000, 50)), rowvar=False)
            false_signals += rmt.denoise(corr, 1000).n_signal
        assert false_signals == 0

    def test_real_spectrum_shape_does_not_pin_the_noise_variance(self):
        """Regression: a density fit of sigma^2 hit its upper bound (1.0) on every
        real window: one dominant market eigenvalue then a smooth decay. The
        estimator must give sigma^2 = 1 - lambda_max / N there instead."""
        values = np.array([15.5, 3.7, 2.2, 1.2] + list(np.linspace(0.9, 0.05, 26)))
        values = values / values.sum() * 30  # a valid trace for N = 30
        sigma2 = rmt.noise_variance(values, 4900)
        assert sigma2 == pytest.approx(1 - values[0] / 30)
        assert sigma2 < 0.6

    @pytest.mark.parametrize("k", [3, 5])
    def test_recovers_three_and_five_factors(self, k):
        rng = np.random.default_rng(k)
        t, n = 500, k * (40 // k)  # equal-size sectors
        market, sectors = rng.normal(size=(t, 1)), rng.normal(size=(t, k))
        membership = np.repeat(np.eye(k), n // k, axis=1)
        corr = np.corrcoef(0.6 * market + 0.7 * sectors @ membership + rng.normal(size=(t, n)),
                           rowvar=False)
        assert rmt.denoise(corr, t).n_signal == k

    def test_tracy_widom_edge_sits_just_above_lambda_plus(self):
        _, hi = rmt.marchenko_pastur_bounds(1.0, 1000 / 50)
        edge = rmt.tracy_widom_edge(1.0, 1000, 50)
        assert hi < edge < hi * 1.05


class TestDenoise:
    def _factor_universe(self, t=500, n=39, seed=1):
        """Market plus three equal sectors: every asset has the same idiosyncratic
        share, which is the setting Marchenko-Pastur describes. The factor
        structure has rank 3 (the market is the sum of the sectors)."""
        rng = np.random.default_rng(seed)
        market = rng.normal(size=(t, 1))
        sectors = rng.normal(size=(t, 3))
        membership = np.repeat(np.eye(3), n // 3, axis=1)
        returns = 0.6 * market + 0.7 * sectors @ membership + rng.normal(size=(t, n))
        return np.corrcoef(returns, rowvar=False), t

    @pytest.mark.parametrize("seed", range(5))
    def test_recovers_the_planted_number_of_factors(self, seed):
        corr, t = self._factor_universe(seed=seed)
        assert rmt.denoise(corr, t).n_signal == 3

    def test_signal_survives_and_noise_is_flattened(self):
        corr, t = self._factor_universe()
        result = rmt.denoise(corr, t)
        # The market eigenvalue is kept exactly; the noise bulk is now flat.
        assert result.denoised_eigenvalues[0] == pytest.approx(result.eigenvalues[0])
        tail = result.denoised_eigenvalues[result.n_signal :]
        assert np.allclose(tail, tail[0])
        # Trace (total variance) is preserved before rescaling.
        assert result.denoised_eigenvalues.sum() == pytest.approx(result.eigenvalues.sum())

    def test_result_is_a_valid_correlation_matrix(self):
        corr, t = self._factor_universe()
        check = check_psd(rmt.denoise(corr, t).correlation)
        assert check.is_psd and check.unit_diagonal

    def test_small_universe_is_warned(self):
        rng = np.random.default_rng(2)
        corr = np.corrcoef(rng.normal(size=(500, 8)), rowvar=False)
        assert any("Only 8 assets" in w for w in rmt.denoise(corr, 500).warnings)

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="more observations than assets"):
            rmt.denoise(np.eye(10), 8)
