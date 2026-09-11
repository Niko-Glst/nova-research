"""Fundamental analysis on company financials, benchmarked against sector peers.

Pulls fundamentals via yfinance and produces a structured fundamental read for a
symbol: absolute metrics (EPS, margins, growth, leverage) plus a relative read
that compares valuation against peers in the same niche. A P/E of 36 means
nothing on its own -- it only becomes a signal next to what the rest of the
industry trades at, which is what `compare_to_peers` provides.

Peer groups are declared in config/allowed_sources.yaml under `peer_groups`.
Research output only -- nothing here places an order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

import yfinance as yf

from common.rate_limit import get_limiter
from config.settings import load_allowed_sources

# Raw yfinance `info` keys this module reads, mapped to the names used downstream.
_RAW_FIELDS = {
    "trailingEps": "trailing_eps",
    "forwardEps": "forward_eps",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "priceToSalesTrailing12Months": "price_to_sales",
    "pegRatio": "peg_ratio",
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "totalRevenue": "total_revenue",
    "freeCashflow": "free_cashflow",
    "profitMargins": "profit_margin",
    "grossMargins": "gross_margin",
    "operatingMargins": "operating_margin",
    "returnOnEquity": "return_on_equity",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "beta": "beta",
}

# Valuation multiples: a lower value is the cheaper read when comparing peers.
_LOWER_IS_CHEAPER = ("trailing_pe", "forward_pe", "price_to_book", "price_to_sales", "peg_ratio")

# Quality metrics: a higher value is the better read when comparing peers.
_HIGHER_IS_BETTER = (
    "profit_margin",
    "operating_margin",
    "revenue_growth",
    "earnings_growth",
    "return_on_equity",
)

# How far from the peer median counts as a real discount/premium rather than noise.
_VALUATION_TOLERANCE = 0.15


@dataclass(frozen=True)
class PeerComparison:
    """How a symbol's valuation stacks up against its niche."""

    niche: str
    peers_used: list[str]
    is_sector_fallback: bool  # True when no niche-level group matched
    symbol_metrics: dict[str, float]
    peer_medians: dict[str, float]
    relative: dict[str, float]  # metric -> symbol / peer_median
    verdict: str  # "cheap" / "expensive" / "in-line" / "unknown"
    notes: str


@dataclass(frozen=True)
class FundamentalRead:
    symbol: str
    metrics: dict[str, float]
    signal: str  # "undervalued" / "overvalued" / "fair"
    notes: str
    reasons: list[str] = field(default_factory=list)
    peer_comparison: PeerComparison | None = None


def _normalize_key(text: str) -> str:
    """Turn a yfinance sector/industry label into a config key.

    "Software - Infrastructure" -> "software_infrastructure"
    """
    collapsed = re.sub(r"[^a-z0-9]+", "_", (text or "").lower())
    return re.sub(r"_+", "_", collapsed).strip("_")


def fetch_fundamentals(symbol: str) -> dict[str, float]:
    """Fetch raw fundamental data for a symbol, rate-limited.

    Returns only the fields yfinance actually provided, converted to float.
    Missing fields are omitted rather than defaulted -- a missing P/E and a P/E
    of 0 mean very different things.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    get_limiter("yfinance").wait()
    info = yf.Ticker(symbol.strip().upper()).info or {}

    if not info.get("symbol") and not info.get("shortName"):
        raise ValueError(
            f"No fundamental data returned for {symbol!r}. Check the ticker spelling."
        )

    out: dict[str, float] = {}
    for raw_key, name in _RAW_FIELDS.items():
        value = info.get(raw_key)
        if value is None:
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue

    # Sector/industry are strings, kept under private keys so they travel with
    # the metrics dict without polluting the numeric output.
    out["_sector"] = info.get("sector") or ""
    out["_industry"] = info.get("industry") or ""
    out["_name"] = info.get("shortName") or symbol.strip().upper()
    return out


def compute_ratios(fundamentals: dict[str, float]) -> dict[str, float]:
    """Derive additional ratios that yfinance does not supply directly."""
    derived: dict[str, float] = {}

    eps = fundamentals.get("trailing_eps")
    forward_eps = fundamentals.get("forward_eps")
    market_cap = fundamentals.get("market_cap")
    revenue = fundamentals.get("total_revenue")
    fcf = fundamentals.get("free_cashflow")

    # Expected EPS growth: what the forward estimate implies versus trailing.
    if eps and forward_eps and eps > 0:
        derived["eps_growth_implied"] = (forward_eps / eps - 1.0) * 100.0

    # Earnings yield -- the inverse of P/E, directly comparable to a bond yield.
    pe = fundamentals.get("trailing_pe")
    if pe and pe > 0:
        derived["earnings_yield_pct"] = (1.0 / pe) * 100.0

    if market_cap and market_cap > 0:
        if fcf:
            derived["fcf_yield_pct"] = (fcf / market_cap) * 100.0
        if revenue:
            derived["revenue_per_market_cap"] = revenue / market_cap

    return derived


def _resolve_peers(symbol: str, sector: str, industry: str) -> tuple[list[str], str, bool]:
    """Find the peer tickers for a symbol.

    Returns (peers, niche_label, is_sector_fallback). Prefers an industry-level
    group; falls back to the broader sector list when the niche is not declared.
    """
    config = load_allowed_sources()
    groups = config.get("peer_groups") or {}
    sector_defaults = config.get("default_peers_by_sector") or {}
    max_peers = int(config.get("max_peers", 6))

    industry_key = _normalize_key(industry)
    sector_key = _normalize_key(sector)

    peers = groups.get(industry_key)
    niche = industry or "unknown"
    fallback = False

    if not peers:
        peers = sector_defaults.get(sector_key)
        niche = sector or "unknown"
        fallback = True

    if not peers:
        return [], niche, True

    # Never compare a symbol against itself.
    peers = [p for p in peers if p.upper() != symbol.upper()][:max_peers]
    return peers, niche, fallback


def _is_comparable(name: str, value: float | None) -> bool:
    """Whether a peer's metric value belongs in the median.

    Valuation multiples are only meaningful when positive: a negative P/E means
    the company is loss-making, and averaging it in would drag the peer median
    toward a number that describes nothing.
    """
    if value is None:
        return False
    if name in _LOWER_IS_CHEAPER:
        return value > 0
    return True


def compare_to_peers(symbol: str, metrics: dict[str, float]) -> PeerComparison:
    """Compare a symbol's valuation metrics against the median of its niche.

    Each peer costs one rate-limited request; peers that fail to fetch are
    skipped rather than aborting the comparison.
    """
    symbol = symbol.strip().upper()
    sector = str(metrics.get("_sector", ""))
    industry = str(metrics.get("_industry", ""))
    peers, niche, fallback = _resolve_peers(symbol, sector, industry)

    if not peers:
        return PeerComparison(
            niche=niche,
            peers_used=[],
            is_sector_fallback=True,
            symbol_metrics={},
            peer_medians={},
            relative={},
            verdict="unknown",
            notes=(
                f"No peer group is declared for industry {industry!r} / sector "
                f"{sector!r}. Add one under `peer_groups` in "
                f"config/allowed_sources.yaml to enable niche comparison."
            ),
        )

    comparable = list(_LOWER_IS_CHEAPER) + list(_HIGHER_IS_BETTER)
    collected: dict[str, list[float]] = {name: [] for name in comparable}
    used: list[str] = []

    for peer in peers:
        try:
            peer_metrics = fetch_fundamentals(peer)
        except Exception:
            # A peer that fails to fetch shouldn't sink the whole comparison.
            continue
        used.append(peer)
        for name in comparable:
            value = peer_metrics.get(name)
            if _is_comparable(name, value):
                collected[name].append(float(value))

    if not used:
        return PeerComparison(
            niche=niche,
            peers_used=[],
            is_sector_fallback=fallback,
            symbol_metrics={},
            peer_medians={},
            relative={},
            verdict="unknown",
            notes=f"All {len(peers)} peer lookups failed for {niche}.",
        )

    peer_medians = {name: median(vals) for name, vals in collected.items() if vals}
    symbol_metrics = {
        name: float(metrics[name]) for name in comparable if metrics.get(name) is not None
    }

    relative: dict[str, float] = {}
    unpriceable: list[str] = []
    for name, med in peer_medians.items():
        own = symbol_metrics.get(name)
        if own is None or not med:
            continue
        # The same rule that excludes a loss-making peer applies to the symbol
        # itself: a negative P/E is not a discount, it is an absent multiple.
        # Ratios like -0.93x would otherwise read as "cheaper than peers".
        if not _is_comparable(name, own):
            unpriceable.append(name)
            continue
        relative[name] = own / med

    # Verdict from the valuation multiples only: below the peer median on most
    # of them reads cheap, above reads expensive. Quality metrics are reported
    # but deliberately excluded here -- they say whether a company is good, not
    # whether it is cheap.
    votes = [
        relative[name]
        for name in _LOWER_IS_CHEAPER
        if name in relative and relative[name] > 0
    ]
    if not votes:
        verdict = "unknown"
    else:
        cheap = sum(1 for v in votes if v < 1 - _VALUATION_TOLERANCE)
        rich = sum(1 for v in votes if v > 1 + _VALUATION_TOLERANCE)
        verdict = "cheap" if cheap > rich else "expensive" if rich > cheap else "in-line"

    scope = "sector (no niche group declared)" if fallback else "niche"
    notes = (
        f"Compared against {len(used)} {scope} peers: {', '.join(used)}. "
        f"Valuation reads {verdict} versus the peer median."
    )
    if unpriceable:
        notes += (
            f" Excluded {', '.join(sorted(unpriceable))} from the comparison: "
            f"the company's own value is negative, so the multiple is undefined "
            f"rather than cheap."
        )

    return PeerComparison(
        niche=niche,
        peers_used=used,
        is_sector_fallback=fallback,
        symbol_metrics=symbol_metrics,
        peer_medians=peer_medians,
        relative=relative,
        verdict=verdict,
        notes=notes,
    )


def _classify(metrics: dict[str, float], peers: PeerComparison | None) -> tuple[str, list[str]]:
    """Turn fundamentals plus the peer read into a stance and its reasons."""
    reasons: list[str] = []
    score = 0

    eps = metrics.get("trailing_eps")
    if eps is not None:
        if eps > 0:
            reasons.append(f"Trailing EPS is positive at {eps:.2f}.")
        else:
            score -= 1
            reasons.append(f"Trailing EPS is negative at {eps:.2f} (unprofitable).")

    implied = metrics.get("eps_growth_implied")
    if implied is not None:
        if implied > 5:
            score += 1
            reasons.append(f"Forward EPS implies {implied:+.1f}% earnings growth.")
        elif implied < -5:
            score -= 1
            reasons.append(f"Forward EPS implies {implied:+.1f}% earnings decline.")

    margin = metrics.get("profit_margin")
    if margin is not None:
        if margin > 0.15:
            score += 1
            reasons.append(f"Healthy net margin of {margin * 100:.1f}%.")
        elif margin < 0:
            score -= 1
            reasons.append(f"Negative net margin of {margin * 100:.1f}%.")

    growth = metrics.get("revenue_growth")
    if growth is not None:
        if growth > 0.10:
            score += 1
            reasons.append(f"Revenue growing {growth * 100:.1f}% year over year.")
        elif growth < 0:
            score -= 1
            reasons.append(f"Revenue shrinking {growth * 100:.1f}% year over year.")

    leverage = metrics.get("debt_to_equity")
    if leverage is not None and leverage > 200:
        score -= 1
        reasons.append(f"High leverage: debt/equity of {leverage:.0f}%.")

    # The peer read carries double weight: relative valuation is the whole point
    # of comparing within a niche.
    if peers is not None:
        if peers.verdict == "cheap":
            score += 2
            reasons.append(f"Trades at a discount to {peers.niche} peers.")
        elif peers.verdict == "expensive":
            score -= 2
            reasons.append(f"Trades at a premium to {peers.niche} peers.")
        elif peers.verdict == "in-line":
            reasons.append(f"Valuation is in line with {peers.niche} peers.")

    if score >= 2:
        return "undervalued", reasons
    if score <= -2:
        return "overvalued", reasons
    return "fair", reasons


def analyze(symbol: str, include_peers: bool = True) -> FundamentalRead:
    """Run the full FA pipeline for a symbol.

    Set include_peers=False to skip the peer comparison (faster: it avoids one
    rate-limited request per peer).
    """
    symbol = symbol.strip().upper()
    raw = fetch_fundamentals(symbol)
    metrics = {**raw, **compute_ratios(raw)}

    peer_comparison = compare_to_peers(symbol, metrics) if include_peers else None
    signal, reasons = _classify(metrics, peer_comparison)

    industry = raw.get("_industry") or "unknown industry"
    sector = raw.get("_sector") or "unknown sector"
    notes = f"{symbol} ({industry}, {sector}): fundamentals read {signal}."
    if peer_comparison is not None:
        notes += " " + peer_comparison.notes

    # Strip the private string fields before handing metrics downstream.
    numeric = {k: v for k, v in metrics.items() if not k.startswith("_")}

    return FundamentalRead(
        symbol=symbol,
        metrics=numeric,
        signal=signal,
        notes=notes,
        reasons=reasons,
        peer_comparison=peer_comparison,
    )
