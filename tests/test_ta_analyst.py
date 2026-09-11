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
