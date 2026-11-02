"""Name search for listings with no news under their ticker (BESI.AS had 0 articles).

A faked Yahoo search only; nothing here touches the network.
"""

from __future__ import annotations

import pytest

from agents import news_sources as ns
from agents import sentiment_analyst as sa

ALLOWED = ["finance.yahoo.com", "yahoo"]


def _article(title, related=(), link="https://finance.yahoo.com/a", publisher="Investing.com"):
    return {"title": title, "relatedTickers": list(related), "link": link,
            "publisher": publisher, "providerPublishTime": 1793106044}


SEARCH_RESULTS = [
    _article("Besi launches €100 million buyback as Q3 orders beat estimates"),
    _article("Dutch chip firm Besi's shares jump as Applied Materials moves to double stake"),
    _article("BE Semiconductor Shares Jump After Applied Materials Increases Stake",
             publisher="The Wall Street Journal"),
    _article("BofA Prefers These Semicap Top Picks", related=["BESI.AS", "ASML"]),
    _article("European shares mostly higher amid tech gains"),        # not about Besi
    _article("Besi shares slide", link="https://example-blog.net/x", publisher="Some Blog"),
]


class _Search:
    def __init__(self, query, news_count, max_results):
        self.query = query
        self.news = SEARCH_RESULTS


@pytest.fixture
def fake_search(monkeypatch):
    import yfinance as yf

    class _Limiter:
        def wait(self):
            pass

    seen = []

    def search(query, news_count, max_results):
        seen.append(query)
        return _Search(query, news_count, max_results)

    monkeypatch.setattr(yf, "Search", search)
    monkeypatch.setattr(ns, "get_limiter", lambda name: _Limiter())
    return seen


class TestNameSearch:
    def test_keeps_articles_about_the_company(self, fake_search):
        items = ns.fetch_yfinance_search_news("BESI.AS", "BE Semiconductor Industries", ALLOWED)
        titles = [i.title for i in items]
        assert fake_search == ["BE Semiconductor Industries"]
        assert len(titles) == 4  # three name the company, one is tagged with the ticker
        assert not any("European shares" in t for t in titles)  # not about Besi
        assert not any(i.url.startswith("https://example-blog") for i in items)  # allowlist
        assert all(i.provider == "yfinance_search" and i.published for i in items)

    def test_name_terms(self):
        assert ns.name_terms("BESI.AS", "BE Semiconductor Industries") == [
            "BE Semiconductor Industries", "BE Semiconductor", "BESI"]
        assert ns.name_terms("SU.PA", "Schneider Electric") == ["Schneider Electric"]

    def test_ticker_root_needs_three_letters(self):
        assert ns.ticker_root("BESI.AS") == "BESI"
        assert ns.ticker_root("SU.PA") == "" and ns.ticker_root("0522.HK") == ""

    def test_search_is_the_last_resort(self, monkeypatch, fake_search):
        def empty(*args, **kwargs):
            raise ValueError("HTTP 403")

        monkeypatch.setattr(ns, "fetch_finnhub_windowed", empty)
        monkeypatch.setattr(ns, "fetch_tiingo_news", empty)
        monkeypatch.setattr(ns, "fetch_yfinance_news", lambda symbol, allowed: [])
        items, unavailable = ns.fetch_news("BESI.AS", ALLOWED,
                                           company_name="BE Semiconductor Industries")
        assert len(items) == 4 and "yfinance (no headlines matched the allowlist)" in unavailable
        assert not items[0].supports_volume_analysis  # a truncated feed: no volume claim

    def test_no_name_no_search(self, monkeypatch, fake_search):
        monkeypatch.setattr(ns, "fetch_finnhub_windowed", lambda *a, **k: [])
        monkeypatch.setattr(ns, "fetch_tiingo_news", lambda *a, **k: [])
        monkeypatch.setattr(ns, "fetch_yfinance_news", lambda symbol, allowed: [])
        items, _ = ns.fetch_news("XYZ", ALLOWED)
        assert items == [] and fake_search == []

    def test_ticker_news_wins_over_the_search(self, monkeypatch, fake_search):
        monkeypatch.setattr(ns, "fetch_finnhub_windowed", lambda *a, **k: [])
        monkeypatch.setattr(ns, "fetch_tiingo_news", lambda *a, **k: [])
        monkeypatch.setattr(ns, "fetch_yfinance_news", lambda symbol, allowed: ["ticker item"])
        items, _ = ns.fetch_news("BESI.AS", ALLOWED, company_name="BE Semiconductor Industries")
        assert items == ["ticker item"] and fake_search == []


class TestRedditQuery:
    def test_distinctive_root_is_searched_with_the_name(self):
        assert sa.reddit_search_query("BESI.AS", "BE Semiconductor Industries N.V.") == \
            '"BE Semiconductor Industries" OR BESI'

    def test_short_or_numeric_roots_are_not(self):
        assert sa.reddit_search_query("SU.PA", "Schneider Electric S.E.") == '"Schneider Electric"'
        assert sa.reddit_search_query("0522.HK", "ASMPT Limited") == '"ASMPT"'

    def test_us_tickers_unchanged(self):
        assert sa.reddit_search_query("META", "Meta Platforms, Inc.") == "META"

    def test_plain_company_name(self):
        assert sa.plain_company_name("BE Semiconductor Industries N.V.") == \
            "BE Semiconductor Industries"
