"""Market-flow analyst: each scored part on synthetic data, then the whole read offline."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents import flow_analyst as fl

TODAY = date(2026, 10, 21)


def _prices(daily_returns: np.ndarray, start: str = "2024-10-01") -> pd.Series:
    index = pd.bdate_range(start, periods=len(daily_returns) + 1)
    return pd.Series(100 * np.exp(np.concatenate([[0.0], np.cumsum(daily_returns)])), index=index)


def _trending(drift: float, early_vol: float, late_vol: float, n: int = 520, seed: int = 0):
    """Steady drift; volatility early_vol for most of the history, late_vol in the last month."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, early_vol)
    vol[-21:] = late_vol
    return _prices(drift + vol * rng.standard_normal(n))


def _cashflow(repurchase=None, issuance=None, quarters: int = 5) -> pd.DataFrame:
    columns = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30",
                              "2025-06-30", "2025-03-31"][:quarters])
    rows = {}
    if repurchase is not None:
        rows[fl.REPURCHASE_ROW] = repurchase
    if issuance is not None:
        rows[fl.ISSUANCE_ROW] = issuance
    return pd.DataFrame(rows, index=columns).T


class TestMeasures:
    def test_trend_returns_and_short_history(self):
        close = pd.Series(np.linspace(100, 200, 300))
        trend = fl.trend_returns(close)
        assert trend[21] == pytest.approx(200 / close.iloc[-22] - 1)
        assert math.isnan(fl.trend_returns(close.iloc[:100])[252])

    def test_volatility_ratio_detects_a_calm_and_a_storm(self):
        assert fl.volatility_ratio(_trending(0.0, 0.02, 0.005)) < 0.5
        assert fl.volatility_ratio(_trending(0.0, 0.01, 0.03)) > 2.0
        assert math.isnan(fl.volatility_ratio(_trending(0.0, 0.01, 0.01, n=200)))


class TestSystematicParts:
    def test_market_up_and_calm_adds_exposure(self):
        part = fl.market_component(_trending(0.002, 0.01, 0.006))
        assert part.score == 1 and "add exposure" in part.reason
        assert part.values["up_horizons"] >= 2

    def test_market_down_and_stormy_cuts_exposure(self):
        part = fl.market_component(_trending(-0.003, 0.01, 0.03))
        assert part.score == -1 and "cut exposure" in part.reason

    def test_market_up_but_stormy_is_no_push(self):
        assert fl.market_component(_trending(0.003, 0.01, 0.03)).score == 0

    def test_market_without_history_is_unknown_not_zero(self):
        part = fl.market_component(_trending(0.001, 0.01, 0.01, n=100))
        assert part.score is None

    def test_stock_volatility_settling_rising_and_flat_discrete(self):
        part = fl.stock_volatility_component(_trending(0, 0.02, 0.01), "X", "discrete")
        assert part.score == 1
        assert part.flag == "vol_estimator: close_to_close"  # a close series has no OHLC
        assert fl.stock_volatility_component(_trending(0, 0.01, 0.02), "X", "discrete").score == -1
        # Exactly constant volatility (alternating +-1%): ratio 1, no change. A random
        # series with constant true volatility can land outside 0.8-1.25 by chance.
        flat = _prices(np.tile([0.01, -0.01], 260))
        assert fl.stock_volatility_component(flat, "X", "discrete").score == 0
        # 21 alternating returns do not split evenly between signs: ratio 1.02, z 0.13.
        assert fl.stock_volatility_component(flat, "X", "continuous").score == pytest.approx(
            0.0, abs=0.2)

    def test_continuous_score_is_minus_clipped_z(self):
        part = fl.stock_volatility_component(_trending(0, 0.02, 0.01), "X", "continuous")
        z = part.values["z"]
        assert part.score == pytest.approx(-max(-1.0, min(1.0, z)))
        assert -1.0 <= part.score <= 1.0 and part.score > 0  # volatility settled


def _balance_sheet(shares: dict[str, float]) -> pd.DataFrame:
    """yfinance-style annual balance sheet: one column per fiscal year-end."""
    return pd.DataFrame({pd.Timestamp(day): [count] for day, count in shares.items()},
                        index=["Ordinary Shares Number"])


def _annual_cashflow(year_end: str, repurchase=None, issuance=None) -> pd.DataFrame:
    rows = {"Free Cash Flow": 1e9}
    if repurchase is not None:
        rows[fl.REPURCHASE_ROW] = repurchase
    if issuance is not None:
        rows[fl.ISSUANCE_ROW] = issuance
    return pd.DataFrame({pd.Timestamp(year_end): rows})


def _yf(shares: dict[str, float], cashflow: pd.DataFrame | None, splits=None):
    return fl.yfinance_issuance_inputs(_balance_sheet(shares), cashflow, splits)


YEARS = ("2024-12-31", "2025-12-31")


class TestIssuance:
    def test_issuer_saturates_at_minus_one_on_the_issuance_side(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 110e6},
                     _annual_cashflow(YEARS[1], repurchase=0.0, issuance=5e8))
        part = fl.issuance_component(inputs, market_cap=1e10)
        assert inputs.net_issuance == pytest.approx(math.log(1.1))
        assert part.score == pytest.approx(-fl.ISSUANCE_SIDE_WEIGHT)
        assert part.flag == fl.REPORTED
        assert part.values["issued_yield"] == pytest.approx(0.05)

    def test_buyback_side_is_capped_at_its_smaller_weight(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 97e6},
                     _annual_cashflow(YEARS[1], repurchase=-3e9))
        part = fl.issuance_component(inputs)
        assert part.score == pytest.approx(fl.BUYBACK_SIDE_WEIGHT)
        assert fl.BUYBACK_SIDE_WEIGHT < fl.ISSUANCE_SIDE_WEIGHT  # owner's rule
        assert part.values["repurchased"] == pytest.approx(3e9)

    def test_small_change_scores_proportionally(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 100.5e6}, _annual_cashflow(YEARS[1], 0.0))
        part = fl.issuance_component(inputs)
        expected = -math.log(1.005) / fl.ISSUANCE_FULL_SCALE * fl.ISSUANCE_SIDE_WEIGHT
        assert part.score == pytest.approx(expected)

    def test_missing_buyback_line_is_assumed_zero_and_still_scored(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 104e6}, _annual_cashflow(YEARS[1]))
        part = fl.issuance_component(inputs)
        assert part.flag == fl.ASSUMED_ZERO and part.score is not None
        assert "assumed_zero" in part.reason

    def test_missing_buyback_line_with_a_falling_count_is_a_conflict(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 97e6}, _annual_cashflow(YEARS[1]))
        part = fl.issuance_component(inputs)
        assert part.flag == fl.DATA_CONFLICT and part.score is None
        assert "disagree" in part.reason

    def test_a_tiny_decline_within_tolerance_is_not_a_conflict(self):
        inputs = _yf({YEARS[0]: 100e6, YEARS[1]: 99.8e6}, _annual_cashflow(YEARS[1]))
        assert fl.issuance_component(inputs).flag == fl.ASSUMED_ZERO

    def test_no_statement_is_unknown_but_the_share_count_still_scores(self):
        # Owner's decision 2026-10-22: share count decides; the evidence is unknown.
        for cashflow in (None, pd.DataFrame(), _annual_cashflow("2023-12-31", -1e9)):
            part = fl.issuance_component(_yf({YEARS[0]: 100e6, YEARS[1]: 105e6}, cashflow))
            assert part.flag == fl.UNKNOWN and part.score is not None

    def test_no_share_counts_drops_the_part(self):
        inputs = _yf({YEARS[1]: 100e6}, _annual_cashflow(YEARS[1], -1e9))
        part = fl.issuance_component(inputs)
        assert part.score is None and "No share counts" in part.reason

    def test_unadjusted_split_is_detected_and_adjusted(self):
        splits = pd.Series([10.0], index=pd.DatetimeIndex(["2025-06-10"]))
        inputs = _yf({YEARS[0]: 2.46e9, YEARS[1]: 24.5e9}, _annual_cashflow(YEARS[1], 0.0),
                     splits)
        assert inputs.net_issuance == pytest.approx(math.log(24.5 / 24.6))
        assert "split" in inputs.note

    def test_already_adjusted_counts_are_left_alone(self):
        # yfinance restates history after a split (NVDA FY2024 reads 24,640M).
        splits = pd.Series([10.0], index=pd.DatetimeIndex(["2025-06-10"]))
        inputs = _yf({YEARS[0]: 24.64e9, YEARS[1]: 24.48e9}, _annual_cashflow(YEARS[1], -1e9),
                     splits)
        assert inputs.net_issuance == pytest.approx(math.log(24.48 / 24.64))
        assert inputs.note == ""

    def test_sec_report_without_repurchase_fact_is_confirmed_zero(self):
        from agents.sec_edgar import AnnualReport

        def report(end, prior, repurchased):
            return AnnualReport("0001-26-1", "10-K", date(2026, 2, 23), date(2025, 12, 31),
                                end, prior, date(2024, 12, 31), repurchased, None)

        issuer = fl.issuance_component(fl.sec_issuance_inputs(report(1.01e9, 0.85e9, None)))
        assert issuer.flag == fl.CONFIRMED_ZERO and issuer.score == pytest.approx(-1.0)
        assert "SEC 10-K filed 2026-02-23" in issuer.reason
        conflict = fl.issuance_component(fl.sec_issuance_inputs(report(0.9e9, 1.0e9, None)))
        assert conflict.flag == fl.DATA_CONFLICT and conflict.score is None
        buyer = fl.issuance_component(fl.sec_issuance_inputs(report(0.9e9, 1.0e9, 5e9)))
        assert buyer.flag == fl.REPORTED and buyer.score == pytest.approx(0.5)

    def test_quarter_returns_line_up_with_the_quarter(self):
        close = pd.Series(np.linspace(100, 160, 700),
                          index=pd.bdate_range("2024-06-03", periods=700))
        quarters = fl.quarterly_buybacks(_cashflow(repurchase=[-1e9] * 5), close)
        q = quarters[0]
        end = pd.Timestamp(q.quarter_end)
        start = end - pd.DateOffset(months=3)
        expected = close.loc[:end].iloc[-1] / close.loc[:start].iloc[-1] - 1
        assert q.price_return == pytest.approx(expected) and q.repurchased == 1e9

    def test_a_blank_issuance_line_does_not_hide_the_quarter(self):
        """GOOGL, 2026-10-30: Yahoo left issuance blank for 3 of 5 quarters, and those
        quarters (one with a $5.5B buyback) disappeared from the report."""
        nan = float("nan")
        quarters = fl.quarterly_buybacks(_cashflow(repurchase=[0, 0, 0, -5.5e9, -11.5e9],
                                                   issuance=[12.4e9, nan, nan, nan, 0.0]))
        assert len(quarters) == 5
        assert quarters[3].repurchased == 5.5e9 and math.isnan(quarters[3].issued)
        both_blank = fl.quarterly_buybacks(_cashflow(repurchase=[nan, -1e9], issuance=[nan, 0.0],
                                                     quarters=2))
        assert len(both_blank) == 1

    def test_unknown_amounts_print_as_na(self):
        from agents import research_report

        assert research_report._format_money(float("nan")) == "n/a"


class TestOptions:
    def _chain(self, strikes, oi, iv):
        return pd.DataFrame({"strike": strikes, "openInterest": oi, "impliedVolatility": iv})

    def test_open_interest_top_strikes_and_atm_vol(self):
        chains = {
            "2026-10-23": (self._chain([9, 10, 11], [500, 900, 100], [0.9, 0.8, 0.85]),
                           self._chain([9, 10], [300, 50], [0.95, 0.9])),
            "2026-11-20": (self._chain([9, 10, 11], [100, 200, 4000], [0.7, 0.6, 1e-5]),
                           self._chain([8, 10], [700, 10], [0.75, 0.65])),
        }
        snap = fl.options_snapshot(chains, last_close=10.2, realised_vol=0.5, today=TODAY)
        assert (snap.call_open_interest, snap.put_open_interest) == (5800, 1060)
        assert snap.put_call_ratio == pytest.approx(1060 / 5800)
        assert snap.top_call_strikes[0] == (11.0, 4100)
        # The 23 Oct expiry is under a week out, so ATM vol comes from 20 Nov.
        assert snap.atm_expiration == "2026-11-20"
        assert snap.atm_implied_vol == pytest.approx((0.6 + 0.65) / 2)

    def test_implausible_implied_vol_is_ignored(self):
        chains = {"2026-11-20": (self._chain([10], [10], [1e-5]), self._chain([10], [5], [9.0]))}
        snap = fl.options_snapshot(chains, 10.0, 0.4, TODAY)
        assert math.isnan(snap.atm_implied_vol)

    def test_no_chains(self):
        assert fl.options_snapshot({}, 10.0, 0.4, TODAY) is None


class TestStance:
    NAMES = ("issuance", "systematic_market", "systematic_stock")

    def _parts(self, scores):
        return tuple(fl.FlowComponent(n, n, s, "") for n, s in zip(self.NAMES, scores))

    @pytest.mark.parametrize("scores, expected", [
        ((1, 1, 0), "bullish"), ((1, 0, 0), "neutral"), ((-1, -1, 1), "neutral"),
        ((-1, -1, None), "bearish"), ((None, 1, 0.5), "bullish"),
        ((None, None, None), "unknown"),
    ])
    def test_mean_of_scored_parts(self, scores, expected):
        signal, _ = fl.classify(self._parts(scores))
        assert signal == expected

    def test_market_part_alone_gives_no_stance(self):
        """A ticker with no data once read 'bullish' on the S&P 500's trend alone."""
        signal, score = fl.classify(self._parts((None, 1, None)))
        assert signal == "unknown" and math.isnan(score)

    def test_rescore_follows_the_same_rule(self):
        from agents import rescore

        market_only = {"market_return_1m": "0.01", "market_return_3m": "0.05",
                       "market_return_12m": "0.15", "market_vol_ratio": "0.7"}
        assert math.isnan(rescore.flows_score(market_only))
        with_stock = {**market_only, "stock_vol_z": "-1.5", "stock_vol_ratio": "0.8"}
        assert math.isfinite(rescore.flows_score(with_stock))


class TestAnalyzeOffline:
    def test_full_read_from_fakes(self, monkeypatch):
        stock = _trending(0.001, 0.02, 0.01, seed=5)
        market = _trending(0.001, 0.01, 0.007, seed=6)
        monkeypatch.setattr(fl, "_wait", lambda: None)
        monkeypatch.setattr(fl, "_prices", lambda s: (market if s == fl.MARKET_PROXY
                                                     else stock).to_frame("Close"))
        chain = SimpleNamespace(
            calls=pd.DataFrame({"strike": [100.0], "openInterest": [10],
                                "impliedVolatility": [0.3]}),
            puts=pd.DataFrame({"strike": [100.0], "openInterest": [20],
                               "impliedVolatility": [0.32]}),
        )
        fake = SimpleNamespace(
            info={"marketCap": 1e11, "currency": "USD", "financialCurrency": "USD",
                  "heldPercentInstitutions": 0.6, "heldPercentInsiders": 0.02},
            quarterly_cashflow=_cashflow(repurchase=[-1e9] * 5),
            balance_sheet=_balance_sheet({YEARS[0]: 100e6, YEARS[1]: 98e6}),
            cashflow=_annual_cashflow(YEARS[1], repurchase=-2e9),
            splits=pd.Series(dtype=float),
            options=("2026-11-20", "2027-06-18"),
            option_chain=lambda expiration: chain,
        )
        monkeypatch.setattr(fl.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(fl.sec_edgar, "cik_for", lambda symbol: None)  # no network
        read = fl.analyze("test", today=TODAY)
        assert read.symbol == "TEST"
        assert [c.name for c in read.components] == ["issuance", "systematic_market",
                                                      "systematic_stock"]
        assert read.component("issuance").flag == fl.REPORTED
        assert read.component("issuance").score > 0
        assert read.signal in ("bullish", "neutral", "bearish")
        assert read.options.expirations == ("2026-11-20",)  # 2027 is beyond 45 days
        assert read.institutions_pct == 0.6
        assert any(r.startswith("Options (not scored)") for r in read.reasons)

    def test_every_source_failing_is_unknown_not_an_error(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("offline")

        monkeypatch.setattr(fl, "_wait", lambda: None)
        monkeypatch.setattr(fl, "_prices", boom)
        monkeypatch.setattr(fl.yf, "Ticker", lambda symbol: SimpleNamespace(
            info={}, quarterly_cashflow=pd.DataFrame(), options=()))
        monkeypatch.setattr(fl.sec_edgar, "cik_for", boom)
        read = fl.analyze("TEST", today=TODAY)
        assert read.signal == "unknown" and math.isnan(read.score)
        assert all(c.score is None for c in read.components)


def test_an_unchanged_share_count_says_so():
    """XFAB.PA, 2026-11-02: 130.6M shares three years running read "shares shrank"."""
    part = fl.issuance_component(fl.IssuanceInputs(
        net_issuance=0.0, year_end=TODAY, source="yfinance annual balance sheet",
        buyback_flag=fl.ASSUMED_ZERO))
    assert "share count unchanged" in part.reason and "shrank" not in part.reason
    assert part.score == 0.0 and math.copysign(1.0, part.score) == 1.0
