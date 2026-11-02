"""Which local index a listing is measured against (config/scoring.yaml `benchmarks`).

A Paris stock is judged against the CAC 40 and an Amsterdam one against the AEX,
not against the S&P 500: what matters for a verdict is whether the stock beat
its own market. The listing is read from the Yahoo suffix (the text after the
last "."); US listings have none. A suffix that is not configured returns None,
and callers say "no benchmark" rather than guess one.

Share-class tickers such as BRK.B carry a suffix that is not an exchange; they
are US listings, so an unconfigured one-letter suffix counts as US (configured
one-letter exchanges, London .L and Tokyo .T, are matched first).

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from config.settings import load_scoring_config

BENCHMARKS: dict[str, str] = {str(k): str(v) for k, v in
                              load_scoring_config()["benchmarks"].items()}


def listing_suffix(symbol: str) -> str:
    """'.AS' for BESI.AS, '' for AAPL, BRK.B (a share class) or ^GSPC."""
    symbol = symbol.strip().upper()
    if symbol.startswith("^") or "." not in symbol:
        return ""
    suffix = symbol[symbol.rfind("."):]
    if suffix in BENCHMARKS:  # a configured exchange, including one-letter .L and .T
        return suffix
    return "" if len(suffix) == 2 else suffix  # ".B" in BRK.B is a US share class


def benchmark_for(symbol: str) -> str | None:
    """The configured local index for `symbol`, or None for an unlisted market."""
    return BENCHMARKS.get(listing_suffix(symbol))
