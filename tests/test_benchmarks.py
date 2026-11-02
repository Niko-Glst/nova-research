"""Regional benchmarks: each listing against its own market."""

from __future__ import annotations

import pytest

from agents import benchmarks


@pytest.mark.parametrize(("symbol", "expected"), [
    ("AAPL", "SPY"), ("BRK.B", "SPY"), ("BESI.AS", "^AEX"), ("su.pa", "^FCHI"),
    ("SAP.DE", "^GDAXI"), ("0522.HK", "^HSI"), ("NESN.SW", "^SSMI"), ("BP.L", "^FTSE"),
    ("7203.T", "^N225"), ("EQNR.OL", None), ("XYZ.ZZ", None),
])
def test_benchmark_for(symbol, expected):
    assert benchmarks.benchmark_for(symbol) == expected


def test_suffixes():
    assert benchmarks.listing_suffix("BESI.AS") == ".AS"
    assert benchmarks.listing_suffix("^GSPC") == ""
    assert benchmarks.listing_suffix("BRK.B") == ""
