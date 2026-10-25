"""Anchored correlation: window sums, the fitted weight, no look-ahead, and the estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlations import anchoring
from correlations.estimators import anchored, estimate_covariance
from correlations.stats import check_psd


def _returns(n_days: int, rho_by_day: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Two assets whose true correlation on each day is rho_by_day[t]."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_days, 2))
    second = rho_by_day * z[:, 0] + np.sqrt(1 - rho_by_day**2) * z[:, 1]
    index = pd.bdate_range("2000-01-03", periods=n_days)
    return pd.DataFrame({"A": z[:, 0] * 0.01, "B": second * 0.015}, index=index)


class TestAnchorPath:
    def test_window_correlations_match_numpy(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(900, 4)) @ rng.normal(size=(4, 4))
        index = pd.bdate_range("2015-01-01", periods=900)
        path = anchoring.anchor_path(values, index, window=252)
        ends = list(index.get_indexer(anchoring.month_ends(index)))
        for i, date in enumerate(path.dates):
            p = index.get_loc(date)
            assert path.recent[i] == pytest.approx(
                np.corrcoef(values[p - 251 : p + 1], rowvar=False), abs=1e-12)
            assert path.long_run[i] == pytest.approx(
                np.corrcoef(values[: p + 1], rowvar=False), abs=1e-12)
            k = ends.index(p)
            if np.isfinite(path.realised[i]).all():
                assert path.realised[i] == pytest.approx(
                    np.corrcoef(values[p + 1 : ends[k + 1] + 1], rowvar=False), abs=1e-12)

    def test_a_constant_correlation_trusts_the_long_run(self):
        # Nothing changes, so the long run is the better estimate: weight near 0.
        returns = _returns(3000, np.full(3000, 0.5))
        result = anchoring.anchored_correlation(returns.to_numpy(), returns.index)
        assert result.n_fit >= anchoring.MIN_FIT_PERIODS
        assert result.weight < 0.3

    def test_persistent_regimes_trust_the_recent_year(self):
        # The truth switches between 0.0 and 0.8 every four years: last year is
        # informative and the all-history average is not.
        regime = (np.arange(4000) // 1000) % 2
        returns = _returns(4000, np.where(regime == 0, 0.0, 0.8), seed=2)
        result = anchoring.anchored_correlation(returns.to_numpy(), returns.index)
        assert result.weight > 0.6

    def test_no_look_ahead(self):
        returns = _returns(1600, np.full(1600, 0.4), seed=3)
        path = anchoring.anchor_path(returns.to_numpy(), returns.index)
        cut = path.dates[-6]
        changed = returns.copy()
        changed.loc[changed.index > cut] *= np.array([1.0, -2.0])
        rerun = anchoring.anchor_path(changed.to_numpy(), changed.index)
        keep = path.dates <= cut
        assert np.allclose(rerun.forecasts[keep], path.forecasts[keep])
        assert np.allclose(rerun.weights[keep], path.weights[keep])

    def test_too_little_history_falls_back_to_the_sample(self):
        returns = _returns(400, np.full(400, 0.4))
        result = anchoring.anchored_correlation(returns.to_numpy(), returns.index)
        assert result.weight == 1.0 and result.n_fit < anchoring.MIN_FIT_PERIODS
        assert result.correlation == pytest.approx(result.recent)
        assert "plain 252-day sample" in result.warnings[0]

    def test_errors_cover_fitted_origins_only(self):
        returns = _returns(2000, np.full(2000, 0.3), seed=4)
        path = anchoring.anchor_path(returns.to_numpy(), returns.index)
        errors = path.offdiagonal_errors("forecast")
        expected = ((path.n_fit >= anchoring.MIN_FIT_PERIODS)
                    & np.isfinite(path.realised).all(axis=(1, 2))).sum()
        assert errors.shape == (expected, 1)


class TestAnchoredEstimator:
    def test_valid_correlation_matrix(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame(rng.normal(size=(1500, 5)) @ rng.normal(size=(5, 5)) * 0.01,
                             index=pd.bdate_range("2010-01-01", periods=1500),
                             columns=list("ABCDE"))
        estimate = anchored(frame)
        assert estimate.is_psd and check_psd(estimate.matrix.to_numpy()).unit_diagonal
        assert 0.0 <= estimate.parameters["weight"] <= 1.0
        assert estimate.parameters["long_run_start"] == "2010-01-01"

    def test_weekly_is_refused(self):
        frame = _returns(500, np.full(500, 0.3))
        estimate = anchored(frame, frequency="weekly")
        assert estimate.matrix.isna().all().all() and "Daily returns only" in estimate.warnings[-1]

    def test_estimate_covariance_passes_the_whole_history(self):
        frame = _returns(1500, np.full(1500, 0.3), seed=6)
        risk = estimate_covariance(frame, "anchored")
        assert risk.correlation.parameters["long_run_start"] == str(frame.index[0].date())
        vol = frame.iloc[-252:].std(ddof=1) * np.sqrt(252)
        assert risk.volatility.to_numpy() == pytest.approx(vol.to_numpy(), rel=1e-12)
        assert risk.window == 252
