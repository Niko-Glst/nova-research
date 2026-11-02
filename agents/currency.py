"""Currencies: minor units, spot conversion and printing amounts in the right symbol.

Yahoo reports each company's market cap in its quote currency (the main unit:
shekels, not the agorot its price is quoted in) and its statements in the
company's reporting currency, and these often differ: Sofwave trades in Tel Aviv
and reports in dollars, BP in London and reports in dollars, TSMC's ADR trades in
dollars and reports in Taiwan dollars.

Conversions use Yahoo's latest spot rate. Its FX *history* is off by a day
(see correlations/fred.py), which does not matter for one current ratio.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import yfinance as yf

from common.rate_limit import get_limiter
from config.settings import load_scoring_config

_CFG = load_scoring_config()["currency"]
MINOR_UNITS = {str(k): (str(v["major"]), float(v["divisor"]))
               for k, v in _CFG["minor_units"].items()}
SYMBOLS = {str(k): str(v) for k, v in _CFG["symbols"].items()}

_rates: dict[str, float] = {"USD": 1.0}  # units per 1 USD, for the life of the process


def major(code: str | None) -> str:
    """The main currency for a Yahoo code: 'ILS' for 'ILA', 'GBP' for 'GBp'."""
    code = (code or "").strip()
    return MINOR_UNITS[code][0] if code in MINOR_UNITS else code


def minor_divisor(code: str | None) -> float:
    """Units per main unit: 100 for agorot and pence, otherwise 1."""
    return MINOR_UNITS[code][1] if code in MINOR_UNITS else 1.0


def _fetch_rate(code: str) -> float | None:
    """Units of `code` per 1 USD, from Yahoo's latest spot quote."""
    get_limiter("yfinance").wait()
    closes = yf.Ticker(f"{code}=X").history(period="5d")["Close"].dropna()
    return float(closes.iloc[-1]) if len(closes) else None


def units_per_usd(code: str) -> float | None:
    """Spot units per dollar, or None when no rate can be had."""
    code = major(code)
    if code not in _rates:
        try:
            rate = _fetch_rate(code)
        except Exception:
            return None
        if not rate or rate <= 0:
            return None
        _rates[code] = rate
    return _rates[code]


def convert(amount: float, source: str, target: str) -> float | None:
    """`amount` from one main currency to another at spot; None without a rate."""
    source, target = major(source), major(target)
    if source == target:
        return amount
    from_rate, to_rate = units_per_usd(source), units_per_usd(target)
    if from_rate is None or to_rate is None:
        return None
    return amount / from_rate * to_rate


def quote_currency(info: dict) -> str:
    """The main currency of a Yahoo info dict's quote; USD when it does not say."""
    return major(info.get("currency")) or "USD"


def report_currency(info: dict) -> str:
    """The currency a company reports in; its quote currency when Yahoo does not say."""
    return major(info.get("financialCurrency")) or quote_currency(info)


def market_cap_usd(info: dict) -> float | None:
    """Yahoo's market cap in dollars; None when it is missing or no rate can be had."""
    cap = info.get("marketCap")
    return convert(float(cap), quote_currency(info), "USD") if cap else None


def prefix(code: str | None) -> str:
    """'$' or '€', or the code itself for currencies without a console-safe symbol."""
    code = major(code) or "USD"
    return SYMBOLS.get(code, f"{code} ")


def money(amount: float, code: str | None = "USD") -> str:
    """A compact amount in its currency: $1.20B, €686.00M, ILS 1.25B; n/a when unknown."""
    if amount != amount:  # NaN: the provider left the line blank
        return "n/a"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(amount) >= threshold:
            return f"{prefix(code)}{amount / threshold:,.2f}{suffix}"
    return f"{prefix(code)}{amount:,.2f}"
