"""SEC EDGAR company facts: parsing, point-in-time filtering, caching. No network."""

from __future__ import annotations

import json
import math
from datetime import date

import pytest

from agents import sec_edgar as sec


def _shares(accn, filed, values, form="10-K"):
    return [{"end": end, "val": val, "accn": accn, "fy": 2025, "fp": "FY", "form": form,
             "filed": filed} for end, val in values]


def _flow(accn, filed, start, end, val, form="10-K"):
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": 2025, "fp": "FY",
            "form": form, "filed": filed}


def _facts(**tags):
    units = {"CommonStockSharesOutstanding": "shares"}
    return {"facts": {"us-gaap": {tag: {"units": {units.get(tag, "USD"): entries}}
                                  for tag, entries in tags.items()}}}


FACTS = _facts(
    CommonStockSharesOutstanding=(
        _shares("A-2025", "2025-02-20", [("2024-12-31", 900.0), ("2023-12-31", 1000.0)])
        + _shares("A-2026", "2026-02-19", [("2025-12-31", 1100.0), ("2024-12-31", 900.0)])
        + _shares("Q-2025", "2025-05-01", [("2025-03-31", 950.0)], form="10-Q")
    ),
    PaymentsForRepurchaseOfCommonStock=[
        _flow("A-2025", "2025-02-20", "2024-01-01", "2024-12-31", 5e9),
        _flow("A-2025", "2025-02-20", "2024-10-01", "2024-12-31", 1e9),  # a quarter: ignored
    ],
    ProceedsFromIssuanceOfCommonStock=[
        _flow("A-2026", "2026-02-19", "2025-01-01", "2025-12-31", 2e9),
    ],
    PaymentsRelatedToTaxWithholdingForShareBasedCompensation=[
        _flow("A-2026", "2026-02-19", "2025-01-01", "2025-12-31", 7e8),
    ],
)


class TestAnnualReports:
    def test_both_counts_come_from_the_same_filing(self):
        reports = sec.annual_reports(FACTS)
        assert [r.accn for r in reports] == ["A-2025", "A-2026"]  # the 10-Q is not annual
        latest = reports[-1]
        assert (latest.shares_end, latest.shares_prior) == (1100.0, 900.0)
        assert latest.net_issuance == pytest.approx(math.log(1100 / 900))

    def test_repurchase_needs_a_full_year_and_its_absence_is_none(self):
        first, latest = sec.annual_reports(FACTS)
        assert first.repurchased == 5e9  # the quarterly fact is not a fiscal year
        assert latest.repurchased is None  # no fact in that 10-K -> confirmed zero
        assert latest.issued == 2e9

    def test_tax_withholding_is_never_a_buyback(self):
        assert "PaymentsRelatedToTaxWithholdingForShareBasedCompensation" in sec.EXCLUDED_TAGS
        assert not set(sec.REPURCHASE_TAGS) & set(sec.EXCLUDED_TAGS)
        assert sec.annual_reports(FACTS)[-1].repurchased is None  # 7e8 withheld, not read

    def test_point_in_time_by_filing_date(self):
        assert sec.annual_reports(FACTS, as_of=date(2025, 2, 19)) == []
        known = sec.latest_annual_report(FACTS, as_of=date(2025, 12, 31))
        assert known.accn == "A-2025"  # the FY2025 10-K is filed only in Feb 2026
        assert sec.latest_annual_report(FACTS, as_of=date(2026, 2, 19)).accn == "A-2026"

    def test_amendment_does_not_replace_the_original(self):
        amended = _facts(CommonStockSharesOutstanding=(
            FACTS["facts"]["us-gaap"]["CommonStockSharesOutstanding"]["units"]["shares"]
            + _shares("A-2026A", "2026-06-01", [("2025-12-31", 1200.0), ("2024-12-31", 900.0)])
        ))
        assert sec.latest_annual_report(amended).accn == "A-2026"


class TestFetching:
    def test_cache_is_used_and_no_network_is_touched(self, tmp_path, monkeypatch):
        calls = []

        def fake(url, headers):
            calls.append((url, headers))
            return json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL"}}).encode()

        monkeypatch.setenv("SEC_USER_AGENT", "test-agent test@example.com")
        monkeypatch.setattr(sec, "get_limiter", lambda name: type("L", (), {"wait": lambda s: None})())
        assert sec.cik_for("aapl", fetcher=fake, cache_dir=tmp_path) == 320193
        assert sec.cik_for("MSFT", fetcher=fake, cache_dir=tmp_path) is None
        assert len(calls) == 1  # second lookup came from the cache
        assert calls[0][1]["User-Agent"] == "test-agent test@example.com"

    def test_hosts_outside_the_allowlist_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not in the SEC allowlist"):
            sec.get_json("https://example.com/x.json", tmp_path / "x.json", 1.0,
                         fetcher=lambda u, h: b"{}")

    def test_a_failed_fetch_falls_back_to_a_stale_copy(self, tmp_path, monkeypatch):
        path = tmp_path / "facts.json"
        path.write_text(json.dumps({"stale": True}), encoding="utf-8")
        monkeypatch.setenv("SEC_USER_AGENT", "test-agent test@example.com")
        monkeypatch.setattr(sec, "get_limiter", lambda name: type("L", (), {"wait": lambda s: None})())

        def down(url, headers):
            raise ConnectionError("offline")

        assert sec.get_json(sec.TICKERS_URL, path, 0.0, fetcher=down) == {"stale": True}
