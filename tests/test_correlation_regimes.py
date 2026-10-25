"""Tests for regime, stress-window, conditional and tail-dependence analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlations import regimes


def _normal_pair(rho: float, n: int, seed: int = 0, scale: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    y = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    index = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"A": z[:, 0] * scale, "B": y * scale}, index=index)


def _switching_market(n_calm=2100, n_wild=900, seed=1):
    """Calm stretch (low vol, corr 0.2) then a volatile one (corr 0.8).

    The calm share (70%) is set just above classify_regimes' 67th-percentile
    split, so its "calm" label falls on the calm stretch.
    """
    calm = _normal_pair(0.2, n_calm, seed, scale=0.005)
    wild = _normal_pair(0.8, n_wild, seed + 1, scale=0.02)
    wild.index = pd.bdate_range(calm.index[-1] + pd.Timedelta(days=1), periods=n_wild)
    frame = pd.concat([calm, wild])
    proxy = (frame["A"] + frame["B"]) / 2
    return frame, proxy


class TestVolatilityRegimes:
    def test_correlation_rises_in_the_volatile_regime(self):
        frame, proxy = _switching_market()
        calm, volatile = regimes.volatility_regimes(frame, proxy, "PROXY")
        assert calm.estimate.pair("A", "B") < 0.4
        assert volatile.estimate.pair("A", "B") > 0.6
        assert "67th percentile" in calm.definition


class TestVix:
    def test_split_and_missing_dates(self):
        frame, _ = _switching_market()
        vix = pd.Series(np.where(np.arange(len(frame)) < 2100, 15.0, 40.0), index=frame.index)
        vix = vix.drop(vix.index[:10])
        high, low = regimes.vix_regimes(frame, vix)
        assert high.estimate.pair("A", "B") > 0.6 > low.estimate.pair("A", "B")
        assert any("VIX missing on 10" in w for w in high.warnings)


class TestStressWindows:
    def test_lower_minimum_with_a_warning(self):
        frame = _normal_pair(0.5, 300)
        start = frame.index[100]
        windows = {
            "short": {"label": "Short", "start": start.date(), "end": frame.index[144].date()},
            "too_short": {"label": "Tiny", "start": start.date(), "end": frame.index[129].date()},
            "outside": {"label": "Old", "start": "1990-01-01", "end": "1990-06-30"},
        }
        short, tiny, outside = regimes.stress_windows(frame, windows)
        assert short.n_observations == 45 and not short.estimate.matrix.isna().all().all()
        assert any("below the usual minimum" in w for w in short.warnings)
        assert tiny.estimate.matrix.isna().all().all()
        assert any("Not covered" in w for w in outside.warnings)


class TestConditionalBenchmark:
    def test_constant_correlation_shows_no_excess(self):
        """Forbes-Rigobon: conditioning on down days lowers a normal pair's
        correlation by itself. The benchmark must absorb that, leaving no excess."""
        frame = _normal_pair(0.5, 20_000, seed=3)
        proxy = frame.mean(axis=1)
        pairs = regimes.conditional_correlations(frame, proxy, "P")
        both_down = next(p for p in pairs if p.condition == "both_down")
        assert both_down.benchmark < 0.4  # truncation lowers it well below 0.5
        assert abs(both_down.excess) < 0.05

    def test_crash_coupling_shows_up_as_excess(self):
        frame = _normal_pair(0.3, 20_000, seed=4)
        rng = np.random.default_rng(9)
        crash = rng.random(len(frame)) < 0.03
        frame.loc[crash, ["A", "B"]] -= 0.04  # joint crash days
        proxy = frame.mean(axis=1)
        pairs = regimes.conditional_correlations(frame, proxy, "P")
        both_down = next(p for p in pairs if p.condition == "both_down")
        assert both_down.excess > 0.1


class TestVectorisedMoments:
    def test_matrix_form_matches_direct_masked_correlation(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(3000, 4)) @ np.array(
            [[1, 0.5, 0.2, 0], [0, 1, 0.3, 0.1], [0, 0, 1, 0.4], [0, 0, 0, 1]]
        )
        corr, count = regimes._pairwise_masked_corr(x, x < 0)
        for a in range(4):
            for b in range(a + 1, 4):
                mask = (x[:, a] < 0) & (x[:, b] < 0)
                direct = np.corrcoef(x[mask, a], x[mask, b])[0, 1]
                assert corr[a, b] == pytest.approx(direct, abs=1e-10)
                assert count[a, b] == mask.sum()


class TestAsymmetry:
    def test_fat_tailed_symmetric_data_shows_no_asymmetry(self):
        """Regression: against the normal benchmark alone, 392 of 435 real pairs
        looked 'more correlated on down days'. A symmetric fat-tailed pair (a
        bivariate t, via a shared random scale) must raise the conditional
        correlations without producing down-versus-up asymmetry."""
        rng = np.random.default_rng(12)
        n = 30_000
        frame = _normal_pair(0.5, n, seed=12)
        scale = np.sqrt(4 / rng.chisquare(4, n))  # t with 4 degrees of freedom
        frame = frame.mul(scale, axis=0)
        conditional = regimes.conditional_correlations(frame, frame.mean(axis=1), "P")
        both_down = next(c for c in conditional if c.condition == "both_down")
        assert both_down.excess > 0.1  # fat tails do beat the normal benchmark...
        (asym,) = regimes.correlation_asymmetry(conditional, frame)
        # ...but not asymmetrically. Regression: Fisher's z called a chance gap of
        # -0.05 on such data significant at p = 3e-6; the bootstrap must not.
        assert not asym.significant_adjusted

    def test_crash_coupling_is_asymmetric_and_significant(self):
        frame = _normal_pair(0.3, 20_000, seed=4)
        rng = np.random.default_rng(9)
        crash = rng.random(len(frame)) < 0.03
        frame.loc[crash, ["A", "B"]] -= 0.04
        conditional = regimes.conditional_correlations(frame, frame.mean(axis=1), "P")
        (asym,) = regimes.correlation_asymmetry(conditional, frame)
        assert asym.difference > 0.1 and asym.significant_adjusted


class TestTailDependence:
    def test_normal_data_matches_its_benchmark(self):
        frame = _normal_pair(0.5, 20_000, seed=5)
        tails = regimes.tail_dependence(frame)
        for t in tails:
            assert abs(t.lower_excess) < 0.05

    def test_joint_crashes_raise_lower_tail_only(self):
        frame = _normal_pair(0.3, 20_000, seed=6)
        rng = np.random.default_rng(10)
        crash = rng.random(len(frame)) < 0.03
        frame.loc[crash, ["A", "B"]] -= 0.04
        q05 = next(t for t in regimes.tail_dependence(frame) if t.quantile == 0.05)
        assert q05.lower_excess > 0.2
        assert q05.upper - q05.upper_benchmark < 0.1

    def test_independent_assets_sit_near_q(self):
        frame = _normal_pair(0.0, 40_000, seed=7)
        q10 = next(t for t in regimes.tail_dependence(frame) if t.quantile == 0.10)
        assert q10.lower == pytest.approx(0.10, abs=0.02)


class TestShiftsAndPipeline:
    def test_diversification_loss_is_flagged(self):
        frame, proxy = _switching_market()
        calm, volatile = regimes.volatility_regimes(frame, proxy, "P")
        shifts = regimes.diversification_shifts(calm.estimate, volatile.estimate, "volatile")
        assert len(shifts) == 1 and shifts[0].change > regimes.DIVERSIFICATION_LOSS

    def test_full_analysis(self):
        frame, proxy = _switching_market()
        vix = pd.Series(np.where(np.arange(len(frame)) < 2100, 15.0, 40.0), index=frame.index)
        windows = {"w": {"label": "W", "start": "2016-01-01", "end": "2016-12-31"}}
        result = regimes.analyze_regimes(frame, proxy, "P", vix, windows)
        names = [r.name for r in result.regimes]
        assert names == ["calm", "volatile", "vix_high", "vix_low", "w"]
        assert result.shifts and result.shifts[0].asset_a == "A"
        assert result.conditional and result.tails
