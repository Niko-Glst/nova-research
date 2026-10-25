"""SEC EDGAR company facts: share counts and buyback/issuance cash flows, point in time.

Two endpoints, both allowlisted in config/allowed_sources.yaml (`sec`), URLs in
config/scoring.yaml (`flows.sec`):

- company_tickers.json maps a ticker to its CIK;
- api/xbrl/companyfacts/CIK##########.json holds every XBRL fact the company
  has filed, each with the filing it came from (`accn`), the form, the period
  and the date it was **filed**.

The `filed` date is what makes this point in time: a fact is only known from
its filing date, never from its period end (which would be look-ahead).

For one annual report (10-K) the module reads:

- shares outstanding at the fiscal year-end and at the prior year-end, both
  from the *same* filing's balance sheet, so both are on the same split basis
  without any split arithmetic;
- the fiscal year's repurchases (PaymentsForRepurchaseOfCommonStock) and
  issuance proceeds (ProceedsFromIssuanceOfCommonStock). Shares withheld for
  employees' taxes on stock compensation
  (PaymentsRelatedToTaxWithholdingForShareBasedCompensation) are never read as
  a buyback. A 10-K with no repurchase fact for the year confirms the year had
  none, as far as the configured tags go.

SEC asks for a descriptive User-Agent with a contact email (SEC_USER_AGENT in
.env) and fewer than 10 requests per second; every request goes through the
shared rate limiter and every response is cached under data/cache/sec.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

from common.files import atomic_write_text
from common.rate_limit import get_limiter
from config.settings import ROOT_DIR, get_sec_user_agent, load_allowed_sources, \
    load_scoring_config

_CFG = load_scoring_config()["flows"]["sec"]

ENABLED = bool(_CFG["enabled"])
TICKERS_URL = str(_CFG["tickers_url"])
COMPANYFACTS_URL = str(_CFG["companyfacts_url"])
TIMEOUT_SECONDS = float(_CFG["timeout_seconds"])
CACHE_DIR = ROOT_DIR / str(_CFG["cache_dir"])
TICKERS_MAX_AGE_HOURS = float(_CFG["cache_max_age_hours"]["tickers"])
FACTS_MAX_AGE_HOURS = float(_CFG["cache_max_age_hours"]["companyfacts"])
ANNUAL_FORMS = tuple(_CFG["annual_forms"])
SHARES_TAGS = tuple(_CFG["shares_tags"])
REPURCHASE_TAGS = tuple(_CFG["repurchase_tags"])
ISSUANCE_TAGS = tuple(_CFG["issuance_tags"])
EXCLUDED_TAGS = tuple(_CFG["excluded_tags"])
ANNUAL_DAYS = tuple(int(d) for d in _CFG["annual_duration_days"])
PRIOR_YEAR_TOLERANCE_DAYS = int(_CFG["prior_year_end_tolerance_days"])

if set(REPURCHASE_TAGS) & set(EXCLUDED_TAGS):
    raise ValueError("A tag cannot be both a repurchase tag and an excluded tag.")

Fetcher = Callable[[str, dict], bytes]


def _http_get(url: str, headers: dict) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _check_allowlisted(url: str) -> None:
    hosts = set((load_allowed_sources().get("sec") or {}).get("hosts") or ())
    host = urlparse(url).hostname or ""
    if host not in hosts:
        raise ValueError(f"{host} is not in the SEC allowlist (config/allowed_sources.yaml).")


def get_json(url: str, cache_path: Path, max_age_hours: float,
             fetcher: Fetcher | None = None) -> dict:
    """A JSON document from the cache when fresh, otherwise fetched politely and cached.

    A failed fetch falls back to a stale cached copy; with no copy, it raises.
    An unreadable cache file counts as no copy, and a failed cache write never
    costs the data just fetched.
    """
    cache_path = Path(cache_path)
    cached = _read_cache(cache_path)
    if cached is not None and (time.time() - cache_path.stat().st_mtime) < max_age_hours * 3600:
        return cached
    _check_allowlisted(url)
    try:
        get_limiter("sec").wait()
        raw = (fetcher or _http_get)(url, {"User-Agent": get_sec_user_agent(),
                                          "Accept": "application/json"})
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        if cached is not None:
            return cached
        raise
    try:
        atomic_write_text(cache_path, json.dumps(data))
    except OSError:
        pass  # the next run fetches again; this one still has its data
    return data


def _read_cache(path: Path) -> dict | None:
    """A cached JSON document, or None when absent or unreadable (e.g. cut short)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cik_for(ticker: str, fetcher: Fetcher | None = None, cache_dir: Path = CACHE_DIR) -> int | None:
    """The CIK SEC files the ticker under, or None (non-US listings have none)."""
    table = get_json(TICKERS_URL, Path(cache_dir) / "company_tickers.json",
                     TICKERS_MAX_AGE_HOURS, fetcher)
    wanted = ticker.strip().upper()
    for row in table.values():
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    return None


def company_facts(cik: int, fetcher: Fetcher | None = None,
                  cache_dir: Path = CACHE_DIR) -> dict:
    url = COMPANYFACTS_URL.format(cik=cik)
    return get_json(url, Path(cache_dir) / f"companyfacts_CIK{cik:010d}.json",
                    FACTS_MAX_AGE_HOURS, fetcher)


# --------------------------------------------------------------------------
# Reading facts
# --------------------------------------------------------------------------


def _entries(facts: dict, tags: tuple[str, ...], unit: str) -> list[dict]:
    out = []
    for tag in tags:
        for entry in (((facts.get("facts") or {}).get("us-gaap") or {}).get(tag) or {}).get(
                "units", {}).get(unit, []):
            out.append({**entry, "tag": tag})
    return out


def _d(text: str) -> date:
    return date.fromisoformat(text)


@dataclass(frozen=True)
class AnnualReport:
    """What one 10-K says about shares and buyback/issuance cash flows."""

    accn: str
    form: str
    filed: date  # the point-in-time date
    fiscal_year_end: date
    shares_end: float | None  # at fiscal_year_end
    shares_prior: float | None  # at the prior year-end, same filing
    prior_year_end: date | None
    repurchased: float | None  # dollars, positive; None = no repurchase fact in this 10-K
    issued: float | None  # dollars, positive; None = no issuance fact in this 10-K

    @property
    def net_issuance(self) -> float | None:
        if self.shares_end and self.shares_prior and self.shares_end > 0 and self.shares_prior > 0:
            import math

            return math.log(self.shares_end / self.shares_prior)
        return None


def _annual_amount(entries: list[dict], accn: str, year_end: date) -> float | None:
    lo, hi = ANNUAL_DAYS
    for e in entries:
        if e.get("accn") != accn or "start" not in e or _d(e["end"]) != year_end:
            continue
        if lo <= (_d(e["end"]) - _d(e["start"])).days <= hi:
            return float(e["val"])
    return None


def annual_reports(facts: dict, as_of: date | None = None) -> list[AnnualReport]:
    """Every annual report known by `as_of` (by filing date), oldest first."""
    shares = [e for e in _entries(facts, SHARES_TAGS, "shares")
              if e.get("form") in ANNUAL_FORMS and e.get("accn")]
    repurchases = _entries(facts, REPURCHASE_TAGS, "USD")
    issuance = _entries(facts, ISSUANCE_TAGS, "USD")
    by_accn: dict[str, list[dict]] = {}
    for e in shares:
        if as_of is None or _d(e["filed"]) <= as_of:
            by_accn.setdefault(e["accn"], []).append(e)
    reports = []
    for accn, items in by_accn.items():
        ends = sorted({_d(e["end"]) for e in items}, reverse=True)
        year_end = ends[0]
        prior = next((d for d in ends[1:]
                      if abs((year_end - d).days - 365) <= PRIOR_YEAR_TOLERANCE_DAYS), None)

        def value_at(day: date | None) -> float | None:
            hit = [float(e["val"]) for e in items if day is not None and _d(e["end"]) == day]
            return hit[0] if hit else None

        reports.append(AnnualReport(
            accn=accn, form=items[0]["form"], filed=min(_d(e["filed"]) for e in items),
            fiscal_year_end=year_end, shares_end=value_at(year_end),
            shares_prior=value_at(prior), prior_year_end=prior,
            repurchased=_annual_amount(repurchases, accn, year_end),
            issued=_annual_amount(issuance, accn, year_end),
        ))
    return sorted(reports, key=lambda r: (r.filed, r.fiscal_year_end))


def latest_annual_report(facts: dict, as_of: date | None = None) -> AnnualReport | None:
    reports = [r for r in annual_reports(facts, as_of) if r.net_issuance is not None]
    if not reports:
        return None
    latest_year = max(r.fiscal_year_end for r in reports)
    # The original filing for the latest year (an amendment filed later is later news).
    return min((r for r in reports if r.fiscal_year_end == latest_year), key=lambda r: r.filed)


def year_end_before(day: date, days: int) -> date:
    return day - timedelta(days=days)
