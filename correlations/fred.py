"""FX reference rates from FRED (Federal Reserve H.10, noon New York).

Why not Yahoo: the EURUSD=X daily history is dated correctly until about 2010
and one day late from 2011 on (SPY's daily return correlates +0.49 with the
same-day bar in 2009 but +0.62 with the next-day bar in 2011). No single shift
fixes both halves, and inferring the shift year by year from the data would
pick whichever lag happens to correlate most, inflating exactly the
correlations this module measures.

The H.10 rates are certified noon buying rates in New York: one consistent
timestamp, correctly dated, back to 1999. Noon is four hours before the US
close, so a small timing gap remains for US assets; weekly returns absorb it,
and fx_timing_check in correlations/data.py measures it on every run.

Only series listed under `fred.series` in config/allowed_sources.yaml can be
fetched (SECURITY.md rule 3). Requests go through the shared rate limiter.
Standard library only: no new dependency.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request

import pandas as pd

from common.rate_limit import get_limiter
from config.settings import load_allowed_sources

PREFIX = "FRED:"

# FRED normally answers in well under a second, but its CDN occasionally holds
# a connection open without sending anything. A short timeout plus a couple of
# retries recovers from that; a persistent failure falls back to the cache in
# correlations.data.load_prices.
REQUEST_TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 3


def is_fred_symbol(symbol: str) -> bool:
    return symbol.upper().startswith(PREFIX)


def fetch_fred_series(symbol: str, period: str = "max") -> pd.DataFrame:
    """Fetch one allowlisted FRED series as a DataFrame with a `Close` column.

    Matches the `fetcher(symbol, period)` signature used by
    correlations.data.load_prices, so FRED series share its cache. `period` is
    accepted for that signature; the full history is always returned.
    """
    series_id = symbol.upper().removeprefix(PREFIX)
    config = load_allowed_sources().get("fred") or {}
    allowed = {str(s).upper() for s in config.get("series") or []}
    if series_id not in allowed:
        raise ValueError(
            f"FRED series {series_id!r} is not on the allowlist. Add it under `fred.series` "
            f"in config/allowed_sources.yaml first (SECURITY.md rule 3)."
        )

    url = f"{config['url']}?{urllib.parse.urlencode({'id': series_id})}"
    request = urllib.request.Request(url, headers={"User-Agent": "nova-research/0.1"})
    last_error: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        get_limiter("fred").wait()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                text = response.read().decode("utf-8")
            return parse_fred_csv(text, series_id)
        except (OSError, TimeoutError) as exc:  # URLError and socket timeouts are OSErrors
            last_error = exc
    raise ConnectionError(
        f"FRED series {series_id} could not be fetched after {MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    )


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse FRED's CSV: a date column, then the series. "." marks a missing day."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected FRED response for {series_id}: {frame.columns.tolist()}")
    date_column = frame.columns[0]
    value_column = series_id if series_id in frame.columns else frame.columns[1]
    out = pd.DataFrame(
        {"Close": pd.to_numeric(frame[value_column], errors="coerce").to_numpy()},
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_column]), name="date"),
    )
    out = out.dropna()
    if out.empty:
        raise ValueError(f"FRED returned no observations for {series_id}.")
    return out
