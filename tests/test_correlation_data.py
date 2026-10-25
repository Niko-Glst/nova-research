"""Tests for correlations.data: universes, listings, cache, FX, alignment, outliers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from correlations import data

CONFIG = {
    "universes": {
        "demo": {
            "symbols": ["SPY", "TLT", "SU.PA"],
            "market_proxy": "SPY",
            "groups": {"SPY": "equity", "TLT": "bonds"},
        }
    },
    "exchanges": {
        "": {"currency": "USD", "region": "americas"},
        ".PA": {"currency": "EUR", "region": "europe"},
        ".SW": {"currency": "CHF", "region": "europe"},
        ".L": {"currency": "GBp", "region": "europe"},
    },
    "symbol_overrides": {"^VIX": {"currency": "USD", "region": "americas"}},
    "minor_units": {"GBp": {"major": "GBP", "divisor": 100}},
    "fx_per_eur": {
        "USD": {"multiply": ["DEXUSEU"]},
        "CHF": {"multiply": ["DEXSZUS", "DEXUSEU"]},
        "GBP": {"multiply": ["DEXUSEU"], "divide": ["DEXUSUK"]},
    },
}


def _labels(currencies, base):
    return {m: r.label for m, r in data.required_fx(currencies, base, CONFIG).items()}


def _history(dates, closes, tz=None) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates)
    if tz:
        index = index.tz_localize(tz)
    return pd.DataFrame({"Close": closes}, index=index)


class TestUniverse:
    def test_from_config(self):
        universe = data.resolve_universe("demo", config=CONFIG)
        assert universe.symbols == ("SPY", "TLT", "SU.PA")
        assert universe.groups["SU.PA"] == "other"
        assert universe.market_proxy == "SPY"

    def test_from_peer_groups(self):
        universe = data.resolve_universe("semiconductors", config=CONFIG)
        assert "NVDA" in universe.symbols
        assert "peer_groups" in universe.source

    def test_override_symbols(self):
        universe = data.resolve_universe("demo", ["gld", "SPY"], config=CONFIG)
        assert universe.symbols == ("GLD", "SPY")

    def test_unknown_universe(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            data.resolve_universe("nope", config=CONFIG)

    def test_real_config_loads(self):
        universe = data.resolve_universe("cross_asset")
        assert {"SPY", "TLT", "GLD"} <= set(universe.symbols)


class TestListing:
    def test_suffixes(self):
        assert data.listing_of("AAPL", CONFIG).currency == "USD"
        assert data.listing_of("SU.PA", CONFIG).region == "europe"
        assert data.listing_of("BARC.L", CONFIG).currency == "GBp"

    def test_unknown_suffix_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError, match="Unknown listing"):
            data.listing_of("7203.T", CONFIG)

    def test_index_needs_an_override(self):
        assert data.listing_of("^VIX", CONFIG).currency == "USD"
        with pytest.raises(ValueError):
            data.listing_of("^STOXX50E", CONFIG)


class TestExchangeDates:
    def test_timezone_keeps_local_date(self):
        # A Paris close at 17:30 local is 16:30 UTC: still the same date. A
        # Tokyo-style midnight stamp must not slide to the previous day.
        series = _history(["2024-03-01 00:00", "2024-03-04 00:00"], [10.0, 11.0], "Asia/Tokyo")
        out, _ = data.to_exchange_dates(series["Close"])
        assert [d.date().isoformat() for d in out.index] == ["2024-03-01", "2024-03-04"]

    def test_duplicates_and_bad_prices(self):
        series = pd.Series(
            [1.0, 1.1, np.nan, 0.0, 1.2],
            index=pd.DatetimeIndex(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        )
        out, notes = data.to_exchange_dates(series)
        assert out.tolist() == [1.1, 1.2]
        assert len(notes) == 3


class TestCache:
    def _fetcher(self, calls):
        def fetch(symbol, period):
            calls.append((symbol, period))
            return _history(pd.bdate_range("2024-01-01", periods=5), [1, 2, 3, 4, 5])

        return fetch

    def test_fresh_cache_is_reused(self, tmp_path):
        calls: list = []
        now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        first = data.load_prices("SPY", cache_dir=tmp_path, fetcher=self._fetcher(calls), now=now)
        second = data.load_prices(
            "SPY", cache_dir=tmp_path, fetcher=self._fetcher(calls), now=now + timedelta(hours=1)
        )
        assert len(calls) == 1 and calls[0] == ("SPY", "max")
        assert not first.from_cache and second.from_cache
        assert second.close.tolist() == first.close.tolist()

    def test_stale_cache_is_refetched(self, tmp_path):
        calls: list = []
        now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        data.load_prices("SPY", cache_dir=tmp_path, fetcher=self._fetcher(calls), now=now)
        later = now + timedelta(hours=data.CACHE_MAX_AGE_HOURS + 1)
        data.load_prices("SPY", cache_dir=tmp_path, fetcher=self._fetcher(calls), now=later)
        assert len(calls) == 2

    def test_refresh_bypasses_cache(self, tmp_path):
        calls: list = []
        now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        data.load_prices("SPY", cache_dir=tmp_path, fetcher=self._fetcher(calls), now=now)
        data.load_prices("SPY", True, tmp_path, self._fetcher(calls), now)
        assert len(calls) == 2

    def test_failed_refetch_falls_back_to_stale_copy(self, tmp_path):
        now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        data.load_prices("SPY", cache_dir=tmp_path, fetcher=self._fetcher([]), now=now)

        def broken(symbol, period):
            raise ConnectionError("offline")

        later = now + timedelta(days=3)
        series = data.load_prices("SPY", cache_dir=tmp_path, fetcher=broken, now=later)
        assert series.from_cache
        assert "Refetch failed" in series.warnings[0]

    def test_failed_fetch_without_cache_raises(self, tmp_path):
        def broken(symbol, period):
            raise ConnectionError("offline")

        with pytest.raises(ConnectionError):
            data.load_prices("SPY", cache_dir=tmp_path, fetcher=broken)

    def test_odd_symbols_get_safe_filenames(self, tmp_path):
        data.load_prices("^VIX", cache_dir=tmp_path, fetcher=self._fetcher([]))
        data.load_prices("EURUSD=X", cache_dir=tmp_path, fetcher=self._fetcher([]))
        names = sorted(p.name for p in tmp_path.glob("*.csv"))
        assert names == ["EURUSD_X.csv", "_VIX.csv"]


class TestCalendarAlignment:
    def test_different_holidays_never_produce_zero_returns(self):
        # Market A closed on the 3rd, market B closed on the 5th.
        a = pd.Series([100.0, 101.0, 103.0, 104.0],
                      index=pd.DatetimeIndex(["2024-01-02", "2024-01-04", "2024-01-05",
                                              "2024-01-08"]))
        b = pd.Series([50.0, 51.0, 52.0, 53.0],
                      index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04",
                                              "2024-01-08"]))
        prices, log = data.align_prices({"A": a, "B": b})
        returns = data.log_returns(prices)

        assert (returns != 0.0).all().all()
        assert list(prices.index.strftime("%m-%d")) == ["01-02", "01-04", "01-08"]
        # B's return over 2nd -> 4th spans the 3rd it traded alone: log(52/50).
        assert returns["B"].iloc[0] == pytest.approx(np.log(52 / 50))
        # A's multi-day return equals the sum of its own daily log returns.
        assert returns["A"].iloc[1] == pytest.approx(np.log(103 / 101) + np.log(104 / 103))
        assert [d.day for d in log.missing_dates["A"]] == [3]
        assert [d.day for d in log.missing_dates["B"]] == [5]
        assert [d.day for d in log.dropped_dates["B"]] == [3]

    def test_young_series_sets_the_start(self):
        old = pd.Series(1.0 + np.arange(10), index=pd.bdate_range("2024-01-01", periods=10))
        young = pd.Series(1.0 + np.arange(5), index=pd.bdate_range("2024-01-08", periods=5))
        _, log = data.align_prices({"OLD": old, "YOUNG": young})
        assert log.limiting_symbol == "YOUNG"
        assert log.common_start == pd.Timestamp("2024-01-08")

    def test_weekly_returns(self):
        index = pd.bdate_range("2024-01-01", periods=15)  # three full weeks
        prices = pd.DataFrame({"A": np.exp(np.arange(15) * 0.01)}, index=index)
        weekly = data.log_returns(prices, "weekly")
        assert len(weekly) == 2
        assert weekly["A"].iloc[0] == pytest.approx(0.05)

    def test_weekly_label_is_the_actual_last_date(self):
        """Regression: a run on Thursday labelled its last week with Friday's date."""
        index = pd.bdate_range("2024-01-01", periods=14)  # ends Thursday 2024-01-18
        prices = pd.DataFrame({"A": np.exp(np.arange(14) * 0.01)}, index=index)
        weekly = data.log_returns(prices, "weekly")
        assert weekly.index[-1] == pd.Timestamp("2024-01-18")


class TestCurrency:
    def _fx(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        return {
            "USD": pd.Series([1.10, 1.20, 1.25], index=index),
            "CHF": pd.Series([0.95, 0.95, 0.95], index=index),
            "GBP": pd.Series([0.85, 0.85, 0.85], index=index),
        }

    def test_usd_asset_into_eur(self):
        close = pd.Series([110.0, 120.0, 125.0], index=pd.bdate_range("2024-01-01", periods=3))
        fx_symbols = _labels(["USD"], "EUR")
        out, log = data.convert_to_base(close, "SPY", "USD", "EUR", self._fx(), fx_symbols, CONFIG)
        assert out.tolist() == pytest.approx([100.0, 100.0, 100.0])
        assert log.fx_series == ("DEXUSEU",)

    def test_eur_asset_into_usd(self):
        close = pd.Series([10.0, 10.0, 10.0], index=pd.bdate_range("2024-01-01", periods=3))
        fx_symbols = _labels(["EUR"], "USD")
        out, _ = data.convert_to_base(close, "SU.PA", "EUR", "USD", self._fx(), fx_symbols, CONFIG)
        assert out.tolist() == pytest.approx([11.0, 12.0, 12.5])

    def test_pence_into_eur(self):
        close = pd.Series([850.0], index=pd.DatetimeIndex(["2024-01-01"]))
        fx_symbols = _labels(["GBp"], "EUR")
        out, _ = data.convert_to_base(close, "X.L", "GBp", "EUR", self._fx(), fx_symbols, CONFIG)
        assert out.iloc[0] == pytest.approx(10.0)  # 8.50 GBP / 0.85 GBP per EUR

    def test_same_currency_is_untouched(self):
        close = pd.Series([1.0, 2.0], index=pd.bdate_range("2024-01-01", periods=2))
        out, log = data.convert_to_base(close, "SPY", "USD", "USD", {}, {}, CONFIG)
        assert out.tolist() == [1.0, 2.0] and log.fx_series == ()

    def test_day_without_fx_quote_is_dropped_not_filled(self):
        close = pd.Series([110.0, 120.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-06"]))
        fx_symbols = _labels(["USD"], "EUR")
        out, log = data.convert_to_base(close, "SPY", "USD", "EUR", self._fx(), fx_symbols, CONFIG)
        assert len(out) == 1
        assert log.dropped_dates == (pd.Timestamp("2024-01-06"),)

    def test_dates_before_fx_history_are_not_missing_quotes(self):
        """Regression: AAPL from 1980 reported 5,828 'missing FX quotes' against an
        EURUSD history that only starts in 2003."""
        close = pd.Series(
            [1.0, 2.0, 110.0], index=pd.DatetimeIndex(["1990-01-02", "1995-01-03", "2024-01-01"])
        )
        fx_symbols = _labels(["USD"], "EUR")
        out, log = data.convert_to_base(close, "SPY", "USD", "EUR", self._fx(), fx_symbols, CONFIG)
        assert log.before_fx_history == 2
        assert log.dropped_dates == ()
        assert len(out) == 1

    def test_missing_fx_configuration(self):
        with pytest.raises(ValueError, match="No FX series"):
            data.required_fx(["JPY"], "EUR", CONFIG)


class TestFxSpikes:
    def _eurusd(self):
        rng = np.random.default_rng(3)
        index = pd.bdate_range("2008-01-01", periods=300)
        return pd.Series(1.3 * np.exp(np.cumsum(rng.normal(0, 0.006, 300))), index=index)

    def test_bad_print_is_removed(self):
        """Regression: Yahoo EURUSD=X shows 1.4918 on 2008-12-08 between 1.2717
        and 1.2926, a fake +16% / -14% pair of days."""
        fx = self._eurusd()
        bad_day = fx.index[150]
        fx[bad_day] = fx.iloc[149] * 1.17
        cleaned, dropped = data.remove_reversal_spikes(fx)
        assert dropped == (bad_day,)
        assert bad_day not in cleaned.index and len(cleaned) == len(fx) - 1

    def test_genuine_move_that_holds_is_kept(self):
        fx = self._eurusd()
        fx.iloc[150:] = fx.iloc[150:] * 1.10  # a real devaluation-style jump that persists
        _, dropped = data.remove_reversal_spikes(fx)
        assert dropped == ()

    def test_ordinary_series_untouched(self):
        _, dropped = data.remove_reversal_spikes(self._eurusd())
        assert dropped == ()


class TestFxRecipes:
    def test_cross_rate_via_usd(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        components = {
            "DEXUSEU": pd.Series([1.10, 1.20, 1.25], index=index),  # USD per EUR
            "DEXSZUS": pd.Series([0.90, 0.80, 0.80], index=index),  # CHF per USD
            "DEXUSUK": pd.Series([1.30, 1.30, 1.25], index=index),  # USD per GBP
        }
        recipes = data.required_fx(["CHF", "GBp"], "EUR", CONFIG)
        chf = data.build_fx_per_eur(recipes["CHF"], components)
        gbp = data.build_fx_per_eur(recipes["GBP"], components)
        assert chf.tolist() == pytest.approx([0.99, 0.96, 1.00])
        assert gbp.tolist() == pytest.approx([1.10 / 1.30, 1.20 / 1.30, 1.0])
        assert recipes["GBP"].label == "DEXUSEU / DEXUSUK"

    def test_holiday_in_one_component_is_missing_not_filled(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        components = {
            "DEXUSEU": pd.Series([1.1, 1.2, 1.3], index=index),
            "DEXSZUS": pd.Series([0.9, 0.9], index=index[[0, 2]]),
        }
        recipe = data.required_fx(["CHF"], "EUR", CONFIG)["CHF"]
        assert len(data.build_fx_per_eur(recipe, components)) == 2


class TestFred:
    def test_parse_csv_with_missing_days(self):
        from correlations import fred

        text = "observation_date,DEXUSEU\n2024-01-01,.\n2024-01-02,1.0956\n2024-01-03,1.0919\n"
        frame = fred.parse_fred_csv(text, "DEXUSEU")
        assert frame["Close"].tolist() == [1.0956, 1.0919]
        assert frame.index[0] == pd.Timestamp("2024-01-02")

    def test_unlisted_series_is_refused(self):
        """SECURITY.md rule 3: only allowlisted FRED series may be fetched."""
        from correlations import fred

        with pytest.raises(ValueError, match="allowlist"):
            fred.fetch_fred_series("FRED:SP500")

    def test_allowlist_covers_every_configured_recipe(self):
        from config.settings import load_allowed_sources

        allowed = set(load_allowed_sources()["fred"]["series"])
        config = data.load_correlation_config()
        needed = {s for e in config["fx_per_eur"].values()
                  for s in e.get("multiply", []) + e.get("divide", [])}
        assert needed <= allowed


class TestFxTiming:
    def _market(self, lag: int):
        """A US asset and an FX rate that reacts to it, with the FX bar dated `lag` days late."""
        rng = np.random.default_rng(8)
        n = 1500
        index = pd.bdate_range("2015-01-01", periods=n)
        us = rng.normal(0, 0.01, n)
        fx_returns = 0.4 * us + rng.normal(0, 0.005, n)
        fx_returns = np.concatenate([np.zeros(lag), fx_returns[: n - lag]])
        to_price = lambda r: pd.Series(np.exp(np.cumsum(r)), index=index)  # noqa: E731
        return to_price(us), to_price(fx_returns)

    def test_late_bars_are_detected(self):
        """Regression: Yahoo EURUSD=X bars from 2011 on are dated one day late
        (SPY corr -0.10 same day, +0.62 with the next bar in 2011). The check
        must flag that pattern."""
        us, fx = self._market(lag=1)
        check = data.fx_timing_check("US", us, "FX", fx)
        assert not check.aligned
        assert "lag +1" in check.note
        assert check.correlations[1] > 0.5

    def test_correctly_dated_bars_pass(self):
        us, fx = self._market(lag=0)
        assert data.fx_timing_check("US", us, "FX", fx).aligned

    def test_intraday_fixing_split_is_not_misdating(self):
        """Regression: the FRED noon rate splits the US session, giving SPY +0.080
        at lag 0 and +0.084 at lag +1. A near-tie must not be called misdated."""
        rng = np.random.default_rng(10)
        n = 3000
        index = pd.bdate_range("2010-01-01", periods=n)
        morning, afternoon = rng.normal(0, 0.007, n), rng.normal(0, 0.007, n)
        us = morning + afternoon
        # Fix at noon: today's FX return sees the morning, tomorrow's the afternoon.
        fx_returns = 0.3 * morning + 0.3 * np.concatenate([[0.0], afternoon[:-1]])
        fx_returns = fx_returns + rng.normal(0, 0.004, n)
        to_price = lambda r: pd.Series(np.exp(np.cumsum(r)), index=index)  # noqa: E731
        check = data.fx_timing_check("US", to_price(us), "FX", to_price(fx_returns))
        assert abs(check.correlations[0] - check.correlations[1]) < 0.1
        assert check.aligned and not check.note

    def test_unrelated_series_are_inconclusive_not_failed(self):
        rng = np.random.default_rng(9)
        index = pd.bdate_range("2015-01-01", periods=800)
        a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=index)
        b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=index)
        check = data.fx_timing_check("A", a, "B", b)
        assert check.aligned and "too weakly related" in check.note


class TestOutliers:
    def test_flags_a_spike(self):
        rng = np.random.default_rng(0)
        returns = pd.DataFrame({"A": rng.normal(0, 0.01, 500)},
                               index=pd.bdate_range("2020-01-01", periods=500))
        returns.iloc[100, 0] = 0.5
        outliers, _ = data.find_outliers(returns)
        assert outliers[0].date == returns.index[100]

    def test_winsorize_clips_tails_only(self):
        values = pd.DataFrame({"A": np.arange(1000.0)})
        clipped = data.winsorize(values, 0.01)
        assert clipped["A"].max() < 999 and clipped["A"].min() > 0
        assert clipped["A"].iloc[500] == 500
