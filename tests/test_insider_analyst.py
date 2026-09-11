"""Tests for insider transaction classification.

The classifier is the whole value of that module: if compensation grants leak
into the buy count, every company looks like its executives are buying. These
cases pin that boundary down, using the exact `Text` strings the provider emits.
"""

from __future__ import annotations

import pytest

from agents import insider_analyst as ia


class TestClassifyTransaction:
    @pytest.mark.parametrize(
        "text",
        [
            "Stock Award(Grant) at price 0.00 per share.",
            "Grant of restricted stock units.",
            "RSU vesting event.",
            "Award at price 0.00 per share.",
        ],
    )
    def test_compensation_is_not_a_purchase(self, text):
        assert ia.classify_transaction(text, 0.0) == ia.GRANT

    @pytest.mark.parametrize(
        "text",
        [
            "Sale at price 498.24 - 505.20 per share.",
            "Sold 1,000 shares.",
            "Disposition at price 100.00 per share.",
        ],
    )
    def test_sales_are_recognized(self, text):
        assert ia.classify_transaction(text, 500_000.0) == ia.SELL

    def test_open_market_purchase_is_recognized(self):
        assert (
            ia.classify_transaction("Purchase at price 100.00 per share.", 250_000.0)
            == ia.BUY
        )

    def test_zero_value_purchase_is_treated_as_compensation(self):
        """A purchase at no cost is a grant whose wording missed the grant patterns."""
        assert ia.classify_transaction("Purchase at price 0.00 per share.", 0.0) == ia.GRANT

    def test_option_exercise_is_separated_from_buying(self):
        assert (
            ia.classify_transaction("Exercise of stock options.", 100_000.0) == ia.OPTION
        )

    def test_grant_wins_over_acquisition_language(self):
        """'Stock Award' describes an acquisition, but it is still compensation."""
        assert (
            ia.classify_transaction("Stock Award(Grant) acquisition of shares", 0.0)
            == ia.GRANT
        )

    def test_unrecognized_text_falls_through_to_other(self):
        assert ia.classify_transaction("Miscellaneous filing", 1000.0) == ia.OTHER

    def test_empty_text_is_handled(self):
        assert ia.classify_transaction("", 0.0) == ia.OTHER


class TestSeniorityDetection:
    @pytest.mark.parametrize(
        "position",
        ["Chief Executive Officer", "CFO", "President", "Chairman of the Board"],
    )
    def test_senior_roles_are_flagged(self, position):
        assert ia._is_senior(position) is True

    @pytest.mark.parametrize("position", ["Director", "Officer", "10% Owner", ""])
    def test_non_senior_roles_are_not_flagged(self, position):
        assert ia._is_senior(position) is False
