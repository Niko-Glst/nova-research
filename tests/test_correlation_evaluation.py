"""Walk-forward evaluation, minimum-variance solvers and estimate_covariance.

Synthetic data with known structure, plus scipy as an independent check of the
long-only solver where it is installed (the KKT test does not need it).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from correlations import evaluation as ev
from correlations.estimators import (
    DEFAULT_COVARIANCE_METHOD,
    estimate_covariance,
    ledoit_wolf_estimate,
    pearson,
)


def _random_covariance(rng: np.random.Generator, n: int, spread: float = 20.0) -> np.ndarray:
    """A sample covariance with volatilities spread over a factor of `spread`."""
    draws = rng.normal(size=(n + int(rng.integers(5, 150)), n))
    vols = np.exp(rng.uniform(0.0, math.log(spread), n)) * 0.002
    return np.corrcoef(draws, rowvar=False) * np.outer(vols, vols)


def _kkt_violation(cov: np.ndarray, weights: np.ndarray) -> float:
    """Largest breach of the long-only optimality conditions, on a unit-trace scale."""
    grad = cov @ weights / (np.trace(cov) / len(cov))
    held = weights > 0
    nu = grad[held].mean()
    stationarity = np.abs(grad[held] - nu).max()
    dual = max(0.0, -(grad[~held] - nu).min()) if (~held).any() else 0.0
    return float(max(stationarity, dual))


def _equicorrelated(n_assets: int = 8, n_days: int = 1100, rho: float = 0.4, seed: int = 3):
    """Returns whose true correlation is rho for every pair, plus the factor."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n_days)
    factor = rng.normal(size=n_days)
    noise = rng.normal(size=(n_days, n_assets))
    vols = np.linspace(0.005, 0.02, n_assets)
    values = (math.sqrt(rho) * factor[:, None] + math.sqrt(1 - rho) * noise) * vols
    columns = [f"A{i}" for i in range(n_assets)]
    returns = pd.DataFrame(values, index=index, columns=columns)
    return returns, pd.Series(factor * 0.01, index=index), {c: "g" for c in columns}


class TestMinVariance:
    def test_unconstrained_matches_the_closed_form(self):
        cov = _random_covariance(np.random.default_rng(0), 6)
        x = np.linalg.solve(cov, np.ones(6))
        result = ev.min_variance(cov, long_only=False)
        assert result.weights == pytest.approx(x / x.sum(), abs=1e-12)

    def test_long_only_equals_unconstrained_when_that_is_already_long(self):
        cov = np.diag([0.04, 0.01, 0.09, 0.02])  # w proportional to 1 / variance
        expected = (1 / np.diag(cov)) / (1 / np.diag(cov)).sum()
        assert ev.min_variance(cov, True).weights == pytest.approx(expected, abs=1e-12)
        assert ev.min_variance(cov, False).weights == pytest.approx(expected, abs=1e-12)

    def test_two_assets_where_the_unconstrained_answer_shorts(self):
        s1, s2, rho = 0.10, 0.20, 0.9
        cov = np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])
        w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
        assert ev.min_variance(cov, False).weights == pytest.approx([w1, 1 - w1], abs=1e-12)
        assert w1 > 1  # short the volatile asset...
        assert ev.min_variance(cov, True).weights == pytest.approx([1.0, 0.0], abs=1e-12)

    def test_long_only_satisfies_the_kkt_conditions(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            cov = _random_covariance(rng, int(rng.integers(2, 31)))
            result = ev.min_variance(cov, True)
            assert result.solved
            assert result.weights.min() >= 0.0
            assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
            assert _kkt_violation(cov, result.weights) < 1e-9

    def test_long_only_matches_scipy_slsqp(self):
        optimize = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(2)
        for _ in range(40):
            n = int(rng.integers(3, 25))
            cov = _random_covariance(rng, n)
            ours = ev.min_variance(cov, True).weights
            reference = optimize.minimize(
                lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                method="SLSQP", bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                options={"ftol": 1e-15, "maxiter": 1000},
            ).x
            # Ours is exact; SLSQP stops at its tolerance, so ours is never worse.
            assert ours @ cov @ ours <= reference @ cov @ reference * (1 + 1e-9)

    def test_undefined_or_singular_covariance_is_refused_with_a_reason(self):
        nan_cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
        result = ev.min_variance(nan_cov)
        assert not result.solved and "undefined" in result.note
        singular = np.ones((3, 3))  # three copies of one asset
        result = ev.min_variance(singular, long_only=False)
        assert not result.solved and "singular" in result.note


class TestDieboldMariano:
    def test_identical_forecasts_have_no_test(self):
        losses = np.random.default_rng(0).random(50)
        statistic, p_value, n = ev.diebold_mariano(losses, losses)
        assert math.isnan(statistic) and math.isnan(p_value) and n == 50

    def test_a_clearly_better_candidate_is_significant_and_positive(self):
        rng = np.random.default_rng(1)
        benchmark = 1.0 + rng.random(120)
        statistic, p_value, _ = ev.diebold_mariano(benchmark, benchmark - 0.2 + 0.01 *
                                                    rng.normal(size=120))
        assert statistic > 0 and p_value < 1e-6

    def test_without_lags_it_is_the_plain_z_test(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=80), rng.normal(size=80)
        d = a - b
        expected = d.mean() / (d.std(ddof=0) / math.sqrt(80))
        statistic, _, _ = ev.diebold_mariano(a, b, lag=0)
        assert statistic == pytest.approx(expected, rel=1e-12)

    def test_size_under_the_null_is_close_to_five_percent(self):
        rng = np.random.default_rng(3)
        rejections = 0
        for _ in range(400):
            # AR(1) loss differences with zero mean: the case Newey-West is for.
            e = rng.normal(size=200)
            d = np.empty(200)
            d[0] = e[0]
            for t in range(1, 200):
                d[t] = 0.5 * d[t - 1] + e[t]
            _, p_value, _ = ev.diebold_mariano(d, np.zeros(200))
            rejections += p_value < 0.05
        assert 0.02 <= rejections / 400 <= 0.11

    def test_too_few_periods(self):
        _, p_value, n = ev.diebold_mariano(np.ones(5), np.zeros(5))
        assert math.isnan(p_value) and n == 5

    def test_lag_rule(self):
        assert ev.newey_west_lag(100) == 4
        assert ev.newey_west_lag(1000) == 6


@pytest.fixture(scope="module")
def result():
    returns, market, groups = _equicorrelated()
    return ev.run_walk_forward(returns, market, groups)


class TestWalkForward:
    def test_monthly_periods_after_the_first_year(self, result):
        assert result.evaluated
        assert result.first_forecast >= pd.Timestamp("2015-12-31")
        assert result.n_periods == len(result.losses) >= ev.MIN_PERIODS
        assert set(result.losses.columns.get_level_values("metric")) == set(ev.METRICS)

    def test_rankings_are_ordered_and_the_benchmark_is_marked(self, result):
        for ranking in result.rankings:
            values = [s.value for s in ranking.scores]
            assert values == sorted(values)
            assert ranking.benchmark.key == ev.BENCHMARK
            assert ranking.benchmark.relative_to_benchmark == pytest.approx(0.0)

    def test_shrinkage_toward_the_true_structure_beats_the_sample(self, result):
        # Every pair truly has correlation 0.4, which is exactly the constant-
        # correlation target: shrinking toward it can only remove noise.
        errors = result.ranking(ev.FORECAST_ERROR)
        lw = errors.score("ledoit_wolf").value
        assert lw < errors.score("sample_252").value < errors.score("sample_60").value

    def test_long_only_weights_are_long(self, result):
        for weights in result.average_weights.values():
            assert min(weights.values()) >= 0.0
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)  # stored to 6 dp

    def test_no_look_ahead(self, result):
        # Change only the last month: no earlier forecast or score may move.
        returns, market, groups = _equicorrelated()
        last = result.losses.index[-1]
        changed = returns.copy()
        changed.loc[changed.index > last] *= -3.0
        rerun = ev.run_walk_forward(changed, market, groups)
        pd.testing.assert_frame_equal(rerun.losses.iloc[:-1], result.losses.iloc[:-1])
        assert not rerun.losses.iloc[-1].equals(result.losses.iloc[-1])

    def test_too_little_history_is_not_ranked(self):
        returns, market, groups = _equicorrelated(n_days=400)
        result = ev.run_walk_forward(returns, market, groups)
        assert not result.evaluated
        assert "at least" in result.warnings[-1]

    def test_without_a_market_factor_the_group_model_is_left_out(self):
        returns, _, _ = _equicorrelated(n_days=900)
        result = ev.run_walk_forward(returns)
        assert "group_factor" not in {c.key for c in result.contestants}
        assert any("group factor" in w for w in result.warnings)

    def test_weekly_returns_are_refused(self):
        returns, _, _ = _equicorrelated(n_days=900)
        result = ev.run_walk_forward(returns, frequency="weekly")
        assert not result.evaluated and "daily" in result.warnings[0]

    def test_rebalance_dates_are_month_ends(self):
        index = pd.bdate_range("2024-01-01", "2024-03-15")
        dates = ev.rebalance_dates(index)
        assert list(dates) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"),
                               pd.Timestamp("2024-03-15")]


class TestEstimateCovariance:
    def test_is_d_c_d_on_the_last_window_annualised(self):
        returns, _, _ = _equicorrelated(n_days=400)
        risk = estimate_covariance(returns, "pearson", window=252)
        tail = returns.iloc[-252:]
        vol = tail.std(ddof=1) * math.sqrt(252)
        assert risk.volatility.to_numpy() == pytest.approx(vol.to_numpy(), rel=1e-12)
        expected = tail.corr().to_numpy() * np.outer(vol, vol)
        assert risk.covariance.to_numpy() == pytest.approx(expected, rel=1e-10)
        assert risk.window == 252 and risk.as_of == returns.index[-1]
        assert risk.correlation.pairs == ()  # matrix only: no 28 significance tests

    def test_default_is_the_walk_forward_choice(self):
        returns, _, _ = _equicorrelated(n_days=400)
        risk = estimate_covariance(returns)
        assert risk.method == DEFAULT_COVARIANCE_METHOD == "ledoit_wolf"
        direct = ledoit_wolf_estimate(returns.iloc[-252:])
        assert risk.correlation.matrix.to_numpy() == pytest.approx(
            direct.matrix.to_numpy(), abs=1e-12)

    def test_short_history_says_so_and_too_short_is_nan(self):
        returns, _, _ = _equicorrelated(n_days=100)
        risk = estimate_covariance(returns)
        assert risk.window == 100 and any("fewer than the 252" in w for w in risk.warnings)
        tiny = estimate_covariance(returns.iloc[:40])
        assert tiny.covariance.isna().all().all()
        assert any("do not size positions" in w for w in tiny.warnings)

    def test_rank_methods_are_refused(self):
        returns, _, _ = _equicorrelated(n_days=300)
        with pytest.raises(ValueError, match="cannot feed a covariance"):
            estimate_covariance(returns, "kendall")

    def test_pearson_keeps_its_pairs_by_default(self):
        returns, _, _ = _equicorrelated(n_days=300)
        assert len(pearson(returns).pairs) == 28
        assert pearson(returns, with_pairs=False).pairs == ()
