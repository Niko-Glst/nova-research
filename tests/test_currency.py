"""Currencies: ratios in one currency, the mega-cap test in dollars, the right symbols.

Spot rates are faked at the values seen on 2026-11-02; nothing touches the network.
Each info dict holds the real Yahoo fields for that stock on that day.
"""

from __future__ import annotations

import pytest

from agents import benchmarks, breadth_analyst, currency, fa_analyst, flow_analyst
from agents.research_report import _format_money, market_cap_text

RATES = {"USD": 1.0, "ILS": 3.1449, "EUR": 0.8932, "GBP": 0.7456, "JPY": 157.905, "TWD": 32.583,
         "CNY": 7.1}

SOFWAVE = {"currency": "ILA", "financialCurrency": "USD", "currentPrice": 3422.0,
           "marketCap": 1245388672, "totalRevenue": 104345000, "bookValue": 1.05,
           "priceToSalesTrailing12Months": 11.935298, "priceToBook": 32.590477,
           "trailingPE": 51.074627, "trailingEps": 0.67, "freeCashflow": 10_000_000}
BP = {"currency": "GBp", "financialCurrency": "USD", "currentPrice": 600.0,
      "marketCap": 92721283072, "totalRevenue": 235135991808, "bookValue": 2.9841275,
      "priceToSalesTrailing12Months": 0.39433044, "priceToBook": 2.010638, "trailingPE": 14.285715}
TSMC = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 402.55,
        "marketCap": 2087815086080, "totalRevenue": 4944817029120, "bookValue": 5.23992,
        "priceToSalesTrailing12Months": 0.4222229, "priceToBook": 76.823685, "trailingPE": 26.09}
TOYOTA = {"currency": "JPY", "financialCurrency": "JPY", "currentPrice": 3051.0,
          "marketCap": 35934875680768, "totalRevenue": 51957024686080, "bookValue": 3150.596,
          "priceToSalesTrailing12Months": 0.6916269, "priceToBook": 0.9683882}
APPLE = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 338.665,
         "marketCap": 4943194357760, "totalRevenue": 466822987776, "bookValue": 7.36,
         "priceToSalesTrailing12Months": 10.587606, "priceToBook": 46.02038}


@pytest.fixture(autouse=True)
def rates(monkeypatch):
    calls = []

    def fetch(code):
        calls.append(code)
        return RATES.get(code)

    monkeypatch.setattr(currency, "_fetch_rate", fetch)
    monkeypatch.setattr(currency, "_rates", {"USD": 1.0})
    return calls


class TestConversion:
    def test_minor_units(self):
        assert currency.major("ILA") == "ILS" and currency.major("GBp") == "GBP"
        assert currency.major("EUR") == "EUR" and currency.major(None) == ""
        assert currency.minor_divisor("ILA") == 100.0 and currency.minor_divisor("USD") == 1.0

    def test_convert_pivots_through_the_dollar(self):
        assert currency.convert(100.0, "USD", "USD") == 100.0
        assert currency.convert(3.1449, "ILS", "USD") == pytest.approx(1.0)
        assert currency.convert(1.0, "GBP", "EUR") == pytest.approx(0.8932 / 0.7456)

    def test_a_rate_is_fetched_once(self, rates):
        currency.convert(1.0, "ILS", "USD")
        currency.convert(2.0, "ILS", "USD")
        assert rates == ["ILS"]

    def test_no_rate_gives_none_and_is_retried(self, monkeypatch, rates):
        assert currency.convert(1.0, "XXX", "USD") is None
        assert currency.convert(1.0, "XXX", "USD") is None
        assert rates == ["XXX", "XXX"]

    def test_a_failing_fetch_gives_none(self, monkeypatch):
        def broken(code):
            raise OSError("offline")

        monkeypatch.setattr(currency, "_fetch_rate", broken)
        assert currency.units_per_usd("ILS") is None

    def test_market_cap_in_dollars(self):
        assert currency.market_cap_usd(SOFWAVE) == pytest.approx(1245388672 / 3.1449)
        assert currency.market_cap_usd(APPLE) == APPLE["marketCap"]
        assert currency.market_cap_usd({}) is None

    def test_money(self):
        assert currency.money(1.2e9) == "$1.20B"
        assert currency.money(686e6, "EUR") == "€686.00M"
        assert currency.money(1.25e9, "ILA") == "ILS 1.25B"  # no console-unsafe symbol
        assert currency.money(float("nan"), "EUR") == "n/a"


class TestRatios:
    def test_sofwave_price_to_sales_and_book(self):
        out = fa_analyst.extract_fundamentals(SOFWAVE, "SOFW.TA")
        assert out["price_to_sales"] == pytest.approx(3.795, abs=0.001)
        assert out["price_to_book"] == pytest.approx((34.22 / 3.1449) / 1.05, abs=0.001)
        assert "11.94" in out["_currency_notes"] and "agorot" not in out["_currency_notes"]
        assert "ILA, 1/100 of a ILS" in out["_currency_notes"]

    def test_the_pe_is_left_alone(self):
        """EPS is in the quote currency: BP's 0.42 is its $0.58 per share in pounds."""
        assert fa_analyst.extract_fundamentals(SOFWAVE, "X")["trailing_pe"] == 51.074627
        assert fa_analyst.extract_fundamentals(BP, "BP.L")["trailing_pe"] == 14.285715

    def test_bp_in_pence(self):
        out = fa_analyst.extract_fundamentals(BP, "BP.L")
        assert out["price_to_sales"] == pytest.approx(0.5289, abs=0.001)
        assert out["price_to_book"] == pytest.approx(6.0 / 0.7456 / 2.9841275, abs=0.001)

    def test_us_listed_foreign_share_keeps_yahoos_book_and_says_so(self):
        out = fa_analyst.extract_fundamentals(TSMC, "TSM")
        assert out["price_to_sales"] == pytest.approx(2087815086080 * 32.583 / 4944817029120)
        assert out["price_to_book"] == 76.823685
        assert "different share basis" in out["_currency_notes"]

    def test_same_currency_is_untouched(self, rates):
        out = fa_analyst.extract_fundamentals(APPLE, "AAPL")
        assert out["price_to_sales"] == 10.587606 and out["price_to_book"] == 46.02038
        assert out["_currency_notes"] == "" and rates == []  # no request at all

    def test_no_rate_keeps_yahoos_figures_and_says_so(self, monkeypatch):
        monkeypatch.setattr(currency, "_fetch_rate", lambda code: None)
        out = fa_analyst.extract_fundamentals(SOFWAVE, "SOFW.TA")
        assert out["price_to_sales"] == 11.935298 and "market_cap_usd" not in out
        assert "no exchange rate" in out["_currency_notes"]

    def test_derived_yields_use_one_currency(self):
        raw = fa_analyst.extract_fundamentals(SOFWAVE, "SOFW.TA")
        ratios = fa_analyst.compute_ratios(raw)
        assert ratios["fcf_yield_pct"] == pytest.approx(10e6 / (1245388672 / 3.1449) * 100)


class TestMegaCapTest:
    def test_toyota_is_not_a_mega_cap_in_dollars(self):
        out = fa_analyst.extract_fundamentals(TOYOTA, "7203.T")
        assert out["market_cap_usd"] == pytest.approx(227.6e9, rel=0.01)
        assert breadth_analyst.classify_stock("Consumer Cyclical", out["market_cap"]) == \
            breadth_analyst.LEADER  # what the raw yen figure wrongly gave
        assert breadth_analyst.classify_stock("Consumer Cyclical", out["market_cap_usd"]) == \
            breadth_analyst.NEUTRAL

    def test_unknown_dollar_value_is_not_a_leader(self):
        assert breadth_analyst.classify_stock("Technology", None) == breadth_analyst.NEUTRAL


class TestDisplay:
    def test_market_cap_shows_its_currency_and_dollars(self):
        out = fa_analyst.extract_fundamentals(SOFWAVE, "SOFW.TA")
        assert market_cap_text(out, "ILS") == "ILS 1.25B (about $396.00M)"
        assert market_cap_text(fa_analyst.extract_fundamentals(APPLE, "AAPL"), "USD") == "$4.94T"

    def test_buyback_text_uses_the_statements_currency(self):
        part = flow_analyst.issuance_component(flow_analyst.IssuanceInputs(
            -0.01, None, "yfinance annual balance sheet", flow_analyst.REPORTED,
            repurchased=82e6, currency="EUR"))
        assert "€82M bought back" in part.reason

    def test_format_money_defaults_to_dollars(self):
        assert _format_money(1.17e9) == "$1.17B" and _format_money(1.17e9, "GBp") == "£1.17B"


def test_tel_aviv_has_a_benchmark():
    assert benchmarks.benchmark_for("SOFW.TA") == "^TA125.TA"
