"""Tests for the finance-tuned sentiment lexicon and aggregation."""

from __future__ import annotations

import pytest

from agents import sentiment_analyst as sa


class TestScoreText:
    def test_positive_market_language(self):
        assert sa.score_text("Apple beats earnings estimates, stock surges") > 0.5

    def test_negative_market_language(self):
        assert sa.score_text("Company misses guidance, shares plunge on downgrade") < -0.5

    def test_negation_flips_polarity(self):
        """'did not beat' must not read as bullish."""
        assert sa.score_text("Stock did not beat expectations") < 0

    def test_negation_only_reaches_within_its_window(self):
        far = "not a single one of the many analysts covering this widely held name beat"
        assert sa.score_text(far) > 0

    @pytest.mark.parametrize(
        "text", ["stock surges", "stock surged", "shares rallied", "company beats estimates"]
    )
    def test_inflected_forms_are_matched(self, text):
        """'surges' must find 'surge' without its own lexicon entry."""
        assert sa.score_text(text) > 0

    @pytest.mark.parametrize("text", ["shares plunging", "losses mounting", "stock dropped"])
    def test_inflected_negative_forms_are_matched(self, text):
        assert sa.score_text(text) < 0

    def test_stemming_does_not_over_match(self):
        """'los angeles' must not be stemmed into 'loss'."""
        assert sa.score_text("a los angeles corporate filing") == 0.0

    def test_text_without_lexicon_terms_is_neutral(self):
        assert sa.score_text("The company filed its quarterly report") == 0.0

    def test_empty_text_is_neutral(self):
        assert sa.score_text("") == 0.0
        assert sa.score_text(None) == 0.0

    def test_score_stays_within_bounds(self):
        extreme = "surge rally soar beat upgrade record strong bullish breakout win"
        assert -1.0 <= sa.score_text(extreme) <= 1.0

    def test_length_does_not_inflate_the_score(self):
        """Averaging, not summing: a long bullish text is not more bullish."""
        short = sa.score_text("stock surges")
        long = sa.score_text("stock surges " + "the company said in a statement " * 20)
        assert long == pytest.approx(short, abs=0.01)


class TestScoreSentiment:
    def test_empty_input_is_neutral(self):
        assert sa.score_sentiment([]) == 0.0

    def test_aggregates_across_mentions(self):
        mentions = [
            {"title": "stock surges on upgrade", "body": ""},
            {"title": "shares rally to record", "body": ""},
        ]
        assert sa.score_sentiment(mentions) > 0.4

    def test_upvotes_increase_influence(self):
        """A heavily upvoted post should pull the average toward itself."""
        mixed = [
            {"title": "stock plunges on fraud probe", "body": "", "weight": 1000},
            {"title": "shares rally", "body": "", "weight": 0},
        ]
        assert sa.score_sentiment(mixed) < 0

    def test_weight_influence_is_capped(self):
        """One viral post must not completely erase the rest of the sample."""
        capped = [
            {"title": "stock plunges on fraud", "body": "", "weight": 10_000_000},
            {"title": "shares rally to record high", "body": "", "weight": 0},
            {"title": "company beats estimates", "body": "", "weight": 0},
            {"title": "stock surges on upgrade", "body": "", "weight": 0},
        ]
        assert sa.score_sentiment(capped) > -0.9

    def test_missing_weight_is_tolerated(self):
        assert sa.score_sentiment([{"title": "stock surges"}]) > 0


class TestRedditSearchQuery:
    def test_us_ticker_is_searched_as_is(self):
        assert sa.reddit_search_query("aapl", "Apple Inc.") == "AAPL"

    def test_suffixed_ticker_is_searched_by_name(self):
        assert sa.reddit_search_query("SU.PA", "Schneider Electric S.E.") == '"Schneider Electric"'

    def test_stacked_legal_suffixes_are_stripped(self):
        assert sa.reddit_search_query("X.DE", "Example Holding AG") == '"Example"'

    def test_suffixed_ticker_without_name_falls_back_to_symbol(self):
        assert sa.reddit_search_query("SU.PA", "") == "SU.PA"


class TestSourceLines:
    def test_reddit_posts_keep_link_score_and_age(self):
        import time

        posts = sa._reddit_headlines([
            {"title": "old post", "body": "", "weight": 3, "source": "r/stocks",
             "url": "https://www.reddit.com/r/stocks/a", "created_utc": time.time() - 5 * 86_400},
            {"title": "shares surge on record profit", "body": "", "weight": 40,
             "source": "r/investing", "url": "https://www.reddit.com/r/investing/b",
             "created_utc": time.time() - 86_400},
        ])
        assert [p.title for p in posts] == ["shares surge on record profit", "old post"]
        assert posts[0].url == "https://www.reddit.com/r/investing/b"
        assert posts[0].score > 0 and posts[1].score == 0
        assert "40 upvotes" in posts[0].source
        assert round(posts[0].age_days) == 1


class TestAllowlistEnforcement:
    def test_unlisted_subreddit_is_refused(self):
        """SECURITY.md rule 3: no fetching from sources outside the allowlist."""
        with pytest.raises(ValueError, match="allowlist"):
            sa.fetch_reddit_mentions("AAPL", ["some_unlisted_subreddit"])
