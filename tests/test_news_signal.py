"""Tests for recency weighting and coverage-volume detection.

All cases build synthetic articles with explicit timestamps, so the windows are
tested against a fixed "now" rather than against whatever the news cycle is
doing today.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agents import news_signal as ns
from agents.news_sources import NewsItem

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _item(days_ago: float | None, title: str = "neutral corporate filing") -> NewsItem:
    published = None if days_ago is None else NOW - timedelta(days=days_ago)
    return NewsItem(
        title=title, body="", published=published, source="reuters.com",
        url="https://reuters.com/x", provider="test",
    )


class TestRecencyWeight:
    def test_fresh_article_keeps_full_weight(self):
        assert ns.recency_weight(0.0) == pytest.approx(1.0)

    def test_half_life_halves_the_weight(self):
        assert ns.recency_weight(7.0, half_life_days=7.0) == pytest.approx(0.5)
        assert ns.recency_weight(14.0, half_life_days=7.0) == pytest.approx(0.25)

    def test_weight_decreases_monotonically(self):
        weights = [ns.recency_weight(d) for d in (0, 1, 3, 7, 30)]
        assert weights == sorted(weights, reverse=True)

    def test_undated_article_is_discounted_not_dropped(self):
        """No timestamp means uncertainty, which is not the same as irrelevance."""
        assert ns.recency_weight(None) == ns.UNDATED_WEIGHT
        assert 0 < ns.UNDATED_WEIGHT < 1

    def test_zero_half_life_disables_decay(self):
        assert ns.recency_weight(100.0, half_life_days=0) == 1.0


class TestAnalyzeVolume:
    def test_spike_is_detected(self):
        """Nine articles in three days against a quiet baseline is a spike."""
        items = [_item(d) for d in (0.2, 0.5, 1.0, 1.2, 2.0, 2.5, 2.8, 1.5, 0.8)]
        items += [_item(d) for d in (10, 20, 25)]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.status == "spike"
        assert volume.ratio > ns.SPIKE_THRESHOLD

    def test_steady_coverage_reads_normal(self):
        items = [_item(d) for d in range(0, 30, 2)]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.status == "normal"
        assert volume.ratio == pytest.approx(1.0, abs=0.4)

    def test_lull_is_detected(self):
        """Dense older coverage with nothing recent is a quiet period."""
        items = [_item(d) for d in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.status == "quiet"
        assert volume.ratio < ns.LULL_THRESHOLD

    def test_single_recent_article_is_not_a_spike(self):
        """One article against a thin baseline is arithmetic, not a news burst.

        Regression: this previously reported a 3.3x spike on a single article.
        """
        items = [_item(0.5), _item(10), _item(20)]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.status == "normal"
        assert "too thin" in volume.notes

    def test_burst_needs_enough_recent_articles(self):
        items = [_item(d) for d in (0.1, 0.2)] + [_item(d) for d in (15, 20, 25)]
        assert ns.analyze_volume(items, now=NOW).status != "spike"

    def test_truncated_feed_refuses_volume_analysis(self):
        """yfinance returns only the newest items, so every symbol would 'spike'.

        Regression: the fallback provider reported a spike for every ticker,
        because a truncated feed has no baseline behind it by construction.
        """
        items = [
            NewsItem("t", "", NOW - timedelta(days=d), "s", "u", "yfinance")
            for d in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 2.5)
        ]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.status == "unknown"
        assert "yfinance" in volume.notes

    def test_full_feed_still_analyzes_volume(self):
        items = [_item(d) for d in (0.1, 0.2, 0.5, 1.0, 2.0, 10, 20, 25)]
        assert ns.analyze_volume(items, now=NOW).status != "unknown"

    def test_too_few_articles_is_unknown(self):
        """A spike computed from two articles is noise, not news."""
        volume = ns.analyze_volume([_item(0.5), _item(1.0)], now=NOW)
        assert volume.status == "unknown"

    def test_undated_articles_do_not_count_toward_volume(self):
        volume = ns.analyze_volume([_item(None) for _ in range(10)], now=NOW)
        assert volume.status == "unknown"

    def test_burst_with_no_baseline_is_a_spike(self):
        """A name nobody was writing about that suddenly is.

        The data must still reach past the recent window, so that an empty
        baseline is a real absence rather than a truncated response.
        """
        items = [_item(0.1), _item(0.5), _item(1.0), _item(2.0), _item(25)]
        volume = ns.analyze_volume(items, recent_window_days=3, now=NOW)
        assert volume.status == "spike"

    def test_data_confined_to_the_recent_window_is_unknown(self):
        """Articles spanning only 2 days cannot be told apart from a capped feed.

        Regression: a provider that caps its response at a few hundred articles
        returns only the newest day or two for a heavily covered name, which
        previously read as an infinite-ratio spike for every such name.
        """
        items = [_item(0.1), _item(0.5), _item(1.0), _item(1.8)]
        volume = ns.analyze_volume(items, recent_window_days=3, now=NOW)
        assert volume.status == "unknown"
        assert "capped" in volume.notes

    def test_baseline_rate_uses_the_span_actually_covered(self):
        """A 9-day response must divide by ~6 baseline days, not the full 27.

        Otherwise the baseline rate is understated and busy names spike falsely.
        """
        items = [_item(d) for d in (0.5, 1.0, 2.0)] + [_item(d) for d in (4, 5, 6, 7, 8)]
        volume = ns.analyze_volume(items, recent_window_days=3, baseline_window_days=30, now=NOW)
        # 5 baseline articles over ~5 covered days, not over 27.
        assert volume.baseline_per_day > 0.5

    def test_baseline_excludes_the_recent_window(self):
        """Otherwise a spike would be measured partly against itself."""
        items = [_item(d) for d in (0.1, 0.2, 0.3, 10, 20)]
        volume = ns.analyze_volume(items, recent_window_days=3, now=NOW)
        assert volume.recent_count == 3
        assert volume.baseline_count == 2

    def test_future_timestamps_are_clamped(self):
        """A provider clock skew must not produce a negative age."""
        items = [_item(-1), _item(0.5), _item(1.0), _item(20)]
        volume = ns.analyze_volume(items, now=NOW)
        assert volume.recent_count == 3


class TestBuildNewsSignal:
    def test_recent_news_dominates_stale_news(self):
        """A bullish story today outweighs a bearish one from a month ago."""
        items = [
            _item(0.1, "stock surges on upgrade after strong results"),
            _item(30, "stock plunges on fraud probe"),
        ]
        signal = ns.build_news_signal(items, _score, now=NOW)
        assert signal.score > 0
        # Without weighting the two would roughly cancel out.
        assert signal.score > signal.unweighted_score

    def test_unweighted_score_is_reported_for_comparison(self):
        items = [_item(0.1, "stock surges"), _item(30, "stock plunges")]
        signal = ns.build_news_signal(items, _score, now=NOW)
        assert signal.unweighted_score == pytest.approx(0.0, abs=0.1)

    def test_effective_sample_reflects_weighting(self):
        """Ten month-old articles are not ten articles' worth of evidence."""
        items = [_item(30) for _ in range(10)]
        signal = ns.build_news_signal(items, _score, now=NOW)
        assert signal.article_count == 10
        assert signal.effective_sample < 3

    def test_empty_input_is_handled(self):
        signal = ns.build_news_signal([], _score, now=NOW)
        assert signal.score == 0.0
        assert signal.article_count == 0
        assert signal.volume.status == "unknown"

    def test_headlines_are_surfaced_by_polarity(self):
        items = [
            _item(0.1, "stock surges on record profit"),
            _item(0.2, "shares plunge on fraud probe"),
            _item(0.3, "company files quarterly report"),
        ]
        signal = ns.build_news_signal(items, _score, now=NOW)
        assert any("surges" in h.title for h in signal.top_positive)
        assert any("plunge" in h.title for h in signal.top_negative)

    def test_headlines_keep_their_link_and_source(self):
        """The report links each headline, so the URL has to survive scoring."""
        items = [_item(0.1, "stock surges on record profit")]
        signal = ns.build_news_signal(items, _score, now=NOW)
        assert signal.top_positive
        headline = signal.top_positive[0]
        assert headline.url == "https://reuters.com/x"
        assert headline.source == "reuters.com"
        assert headline.age_days is not None

    def test_score_stays_within_bounds(self):
        items = [_item(0.0, "surge rally soar beat upgrade record") for _ in range(5)]
        assert -1.0 <= ns.build_news_signal(items, _score, now=NOW).score <= 1.0

    def test_volume_is_not_folded_into_sentiment(self):
        """A spike changes the volume read, never the direction of the score."""
        quiet = [_item(d, "stock surges") for d in (5, 8, 10, 14, 20, 25)]
        spike = [_item(d, "stock surges") for d in (0.1, 0.2, 0.3, 0.4, 10, 20)]
        # `now` must be pinned: the items are dated relative to NOW, so letting
        # this fall back to the wall clock makes the test fail as time passes.
        quiet_signal = ns.build_news_signal(quiet, _score, now=NOW)
        spike_signal = ns.build_news_signal(spike, _score, now=NOW)
        assert spike_signal.score > 0 and quiet_signal.score > 0
        assert spike_signal.volume.status != quiet_signal.volume.status


def _score(text: str) -> float:
    """The real lexicon, injected the way build_news_signal expects."""
    from agents.sentiment_analyst import score_text

    return score_text(text)
