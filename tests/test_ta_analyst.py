"""Tests for the technical analyst's indicator math and classification.

These use synthetic price series rather than live data: an indicator test that
depends on today's market is a test that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents import ta_analyst


def _price_frame(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a list of closing prices."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=index,
    )


class TestRSI:
    def test_unbroken_gains_reads_100(self):
        """With no losing days at all, RSI is 100 by definition."""
        rising = pd.Series(np.arange(100, 150, dtype=float))
        assert ta_analyst._rsi(rising).iloc[-1] == pytest.approx(100.0)

    def test_unbroken_losses_read_near_zero(self):
        falling = pd.Series(np.arange(150, 100, -1, dtype=float))
        assert ta_analyst._rsi(falling).iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_rsi_stays_within_bounds(self):
        rng = np.random.default_rng(seed=42)
        noisy = pd.Series(100 + rng.normal(0, 2, 200).cumsum())
        rsi = ta_analyst._rsi(noisy).dropna()
        assert not rsi.empty
        assert rsi.between(0, 100).all()

    def test_insufficient_history_yields_nan(self):
        """Fewer bars than the period cannot produce a value."""
        short = pd.Series([100.0, 101.0, 102.0])
        assert pd.isna(ta_analyst._rsi(short, period=14).iloc[-1])


class TestMACD:
    def test_signal_line_lags_macd_on_a_trend(self):
        """On a steady rise the MACD line leads its own signal line."""
        rising = pd.Series(np.linspace(100, 200, 120))
        macd, signal = ta_analyst._macd(rising)
        assert macd.iloc[-1] > signal.iloc[-1]

    def test_flat_series_has_no_momentum(self):
        flat = pd.Series([100.0] * 120)
        macd, signal = ta_analyst._macd(flat)
        assert macd.iloc[-1] == pytest.approx(0.0)
        assert signal.iloc[-1] == pytest.approx(0.0)


class TestComputeIndicators:
    def test_returns_expected_keys_with_full_history(self):
        frame = _price_frame(list(np.linspace(100, 180, 300)))
        indicators = ta_analyst.compute_indicators(frame)
        for key in ("close", "sma_50", "sma_200", "rsi_14", "macd", "return_pct"):
            assert key in indicators

    def test_omits_indicators_lacking_history(self):
        """A 60-day series cannot support a 200-day average, so it is left out."""
        frame = _price_frame(list(np.linspace(100, 120, 60)))
        indicators = ta_analyst.compute_indicators(frame)
        assert "sma_50" in indicators
        assert "sma_200" not in indicators

    def test_return_pct_matches_first_and_last_close(self):
        frame = _price_frame([100.0] * 10 + [150.0])
        assert ta_analyst.compute_indicators(frame)["return_pct"] == pytest.approx(50.0)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ta_analyst.compute_indicators(pd.DataFrame({"Close": []}))


class TestClassify:
    def test_uptrend_reads_bullish(self):
        frame = _price_frame(list(np.linspace(100, 200, 300)))
        signal, reasons = ta_analyst._classify(ta_analyst.compute_indicators(frame))
        assert signal == "bullish"
        assert reasons

    def test_downtrend_reads_bearish(self):
        frame = _price_frame(list(np.linspace(200, 100, 300)))
        signal, _ = ta_analyst._classify(ta_analyst.compute_indicators(frame))
        assert signal == "bearish"

    def test_every_vote_is_explained(self):
        """Each directional vote must come with a stated reason."""
        frame = _price_frame(list(np.linspace(100, 200, 300)))
        _, reasons = ta_analyst._classify(ta_analyst.compute_indicators(frame))
        assert len(reasons) >= 4

    def test_no_indicators_reads_neutral(self):
        signal, reasons = ta_analyst._classify({})
        assert signal == "neutral"
        assert reasons == []

    def test_marginal_macd_does_not_outvote_the_trend(self):
        """A histogram of +0.01 on a $16 stock is noise, not momentum.

        Regression: this combination (price below its 50-day average, a barely
        positive histogram) previously read bullish.
        """
        signal, reasons = ta_analyst._classify(
            {
                "close": 16.05,
                "sma_50": 16.48,
                "sma_200": 16.31,
                "rsi_14": 49.8,
                "macd_histogram": 0.01,
            }
        )
        assert signal != "bullish"
        assert any("noise" in r for r in reasons)

    def test_meaningful_macd_still_counts(self):
        indicators = {
            "close": 16.05,
            "sma_50": 16.48,
            "sma_200": 16.31,
            "rsi_14": 49.8,
            "macd_histogram": 0.9,
        }
        _, reasons = ta_analyst._classify(indicators)
        assert any("positive momentum" in r for r in reasons)

    def test_trend_outweighs_oscillators(self):
        """Price below both averages is bearish even with a supportive RSI."""
        signal, _ = ta_analyst._classify(
            {
                "close": 90.0,
                "sma_50": 100.0,
                "sma_200": 110.0,
                "rsi_14": 29.0,  # oversold, votes bullish
                "macd_histogram": 0.0,
            }
        )
        assert signal == "bearish"


class TestFetchValidation:
    def test_blank_symbol_is_rejected_before_any_request(self):
        with pytest.raises(ValueError, match="non-empty"):
            ta_analyst.fetch_price_history("   ")


class TestDonchianChannel:
    def test_channel_excludes_the_current_bar(self):
        """Otherwise a new high would always equal its own upper channel."""
        closes = [100.0] * 20 + [150.0]
        frame = _price_frame(closes)
        upper, _ = ta_analyst._donchian(frame["High"], frame["Low"], period=20)
        # The last bar's channel describes the 20 bars before it, all at ~100.
        assert upper.iloc[-1] < 150.0
        assert frame["Close"].iloc[-1] > upper.iloc[-1]

    def test_upper_tracks_the_prior_high(self):
        frame = _price_frame(list(np.linspace(100, 120, 40)))
        upper, lower = ta_analyst._donchian(frame["High"], frame["Low"], period=10)
        assert upper.iloc[-1] > lower.iloc[-1]

    def test_insufficient_history_yields_nan(self):
        frame = _price_frame([100.0, 101.0, 102.0])
        upper, _ = ta_analyst._donchian(frame["High"], frame["Low"], period=20)
        assert pd.isna(upper.iloc[-1])


class TestVolumeRatio:
    def test_steady_volume_reads_about_one(self):
        volume = pd.Series([1_000_000.0] * 40)
        assert ta_analyst._volume_ratio(volume, period=20).iloc[-1] == pytest.approx(1.0)

    def test_surge_is_detected(self):
        volume = pd.Series([1_000_000.0] * 30 + [3_000_000.0])
        assert ta_analyst._volume_ratio(volume, period=20).iloc[-1] == pytest.approx(3.0)

    def test_average_excludes_today(self):
        """Including today would dampen exactly the spike being measured."""
        volume = pd.Series([1_000_000.0] * 20 + [10_000_000.0])
        ratio = ta_analyst._volume_ratio(volume, period=20).iloc[-1]
        assert ratio == pytest.approx(10.0)

    def test_zero_average_does_not_divide_by_zero(self):
        volume = pd.Series([0.0] * 20 + [500.0])
        assert pd.isna(ta_analyst._volume_ratio(volume, period=20).iloc[-1])


class TestWeightOrdering:
    def test_macd_is_the_smallest_vote(self):
        """MACD lags and flips on small moves, so it must not outrank trend."""
        assert ta_analyst.WEIGHT_MACD < ta_analyst.WEIGHT_RSI
        assert ta_analyst.WEIGHT_MACD < ta_analyst.WEIGHT_VOLUME
        assert ta_analyst.WEIGHT_MACD < ta_analyst.WEIGHT_DONCHIAN

    def test_volume_and_breakout_outrank_oscillators(self):
        assert ta_analyst.WEIGHT_VOLUME > ta_analyst.WEIGHT_RSI
        assert ta_analyst.WEIGHT_DONCHIAN > ta_analyst.WEIGHT_RSI

    def test_trend_still_outranks_everything(self):
        for weight in (
            ta_analyst.WEIGHT_DONCHIAN,
            ta_analyst.WEIGHT_VOLUME,
            ta_analyst.WEIGHT_RSI,
            ta_analyst.WEIGHT_MACD,
        ):
            assert ta_analyst.WEIGHT_PRICE_VS_SMA >= weight


class TestClassifierWithNewSignals:
    def _base(self) -> dict:
        return {
            "close": 110.0,
            "sma_50": 100.0,
            "sma_200": 95.0,
            "rsi_14": 55.0,
            "macd_histogram": 0.5,
            "donchian_upper_20": 105.0,
            "donchian_lower_20": 90.0,
        }

    def test_breakout_is_reported(self):
        _, reasons = ta_analyst._classify(self._base())
        assert any("broke above" in r for r in reasons)

    def test_breakdown_is_reported(self):
        indicators = self._base()
        indicators.update({"close": 85.0, "sma_50": 100.0})
        _, reasons = ta_analyst._classify(indicators)
        assert any("broke below" in r for r in reasons)

    def test_position_within_range_when_no_breakout(self):
        indicators = self._base()
        indicators["close"] = 97.0
        _, reasons = ta_analyst._classify(indicators)
        assert any("of the way up" in r for r in reasons)

    def test_volume_surge_confirms_an_advance(self):
        indicators = self._base()
        indicators["volume_ratio"] = 2.5
        signal, reasons = ta_analyst._classify(indicators)
        assert signal == "bullish"
        assert any("real participation" in r for r in reasons)

    def test_thin_volume_argues_against_the_move(self):
        """A rally nobody participates in is weaker than the same rally on volume."""
        strong = self._base()
        strong["volume_ratio"] = 2.5
        thin = self._base()
        thin["volume_ratio"] = 0.3

        _, thin_reasons = ta_analyst._classify(thin)
        assert any("lacks conviction" in r for r in thin_reasons)

    def test_volume_surge_on_a_decline_is_bearish(self):
        """Heavy volume is only bullish if price is rising on it."""
        indicators = {
            "close": 85.0,
            "sma_50": 100.0,
            "sma_200": 110.0,
            "rsi_14": 45.0,
            "macd_histogram": -0.5,
            "donchian_upper_20": 120.0,
            "donchian_lower_20": 90.0,
            "volume_ratio": 3.0,
        }
        signal, reasons = ta_analyst._classify(indicators)
        assert signal == "bearish"
        assert any("the decline" in r for r in reasons)

    def test_macd_alone_cannot_flip_the_verdict(self):
        """Regression: MACD used to carry the same weight as the trend checks."""
        bearish_trend = {
            "close": 90.0,
            "sma_50": 100.0,
            "sma_200": 110.0,
            "rsi_14": 50.0,
            "macd_histogram": 5.0,  # strongly positive
            "donchian_upper_20": 120.0,
            "donchian_lower_20": 85.0,
        }
        assert ta_analyst._classify(bearish_trend)[0] == "bearish"
