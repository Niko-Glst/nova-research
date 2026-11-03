"""Fundamental analysis on company financials, benchmarked against sector peers.

Pulls fundamentals via yfinance and produces a structured fundamental read for a
symbol: absolute metrics (EPS, margins, growth, leverage) plus a relative read
that compares valuation against peers in the same niche. A P/E of 36 means
nothing on its own -- it only becomes a signal next to what the rest of the
industry trades at, which is what `compare_to_peers` provides.

Peer groups are declared in config/allowed_sources.yaml under `peer_groups`;
peers chosen for one company live in config/custom_peers.yaml and go first.
Research output only -- nothing here places an order.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median

import yaml
import yfinance as yf

from agents import currency
from common.files import atomic_write_text
from common.rate_limit import get_limiter
from config.settings import ROOT_DIR, load_allowed_sources

CUSTOM_PEERS_PATH = ROOT_DIR / "config" / "custom_peers.yaml"
_CUSTOM_PEERS_HEADER = (
    "# Peers chosen per company, written by research.py when it asks in the terminal.\n"
    "# They go before the groups in allowed_sources.yaml. Edit or remove a line by hand\n"
    "# to change a choice. Comments below this header are lost when research.py saves.\n\n"
)

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
    # Per-ticker values, kept so a chart can show each peer rather than only the
    # median: {metric: {ticker: value}}. The subject's own value lives in
    # symbol_metrics, not here. Defaulted, so it must follow the required fields.
    peer_values: dict[str, dict[str, float]] = field(default_factory=dict)
    is_custom: bool = False  # True when the peers were chosen for this company


@dataclass(frozen=True)
class FundamentalRead:
    symbol: str
    metrics: dict[str, float]
    signal: str  # "undervalued" / "overvalued" / "fair"
    notes: str
    reasons: list[str] = field(default_factory=list)
    peer_comparison: PeerComparison | None = None
    name: str = ""  # company name, e.g. for searching sources that don't use tickers
    sector: str = ""  # yfinance sector label (breadth_analyst groups stocks by it)
    quote_currency: str = "USD"  # main currency of the market cap and the price
    report_currency: str = "USD"  # currency of revenue, cash flow and book value


def _normalize_key(text: str) -> str:
    """Turn a yfinance sector/industry label into a config key.

    "Software - Infrastructure" -> "software_infrastructure"
    """
    collapsed = re.sub(r"[^a-z0-9]+", "_", (text or "").lower())
    return re.sub(r"_+", "_", collapsed).strip("_")


_MARGINS = ("profit_margin", "gross_margin", "operating_margin")


def _repair_margins(out: dict, info: dict) -> list[str]:
    """Drop or recompute margins that yfinance reports as a placeholder 0.0.

    yfinance returns exactly 0.0 when it cannot compute a margin. ImmunityBio
    (IBRX) on 2026-10-17: profitMargins 0.0 against a net loss of $992M on
    $166M of revenue, a real margin near -600%. Read at face value, a deep loss
    passes as break-even, escapes the negative-margin penalty and beats
    loss-making peers on profitability. So:

    - no revenue (missing or zero): every margin is undefined and is dropped;
    - profit margin exactly 0.0: recomputed as net income / revenue when net
      income is known and non-zero, otherwise dropped;
    - gross or operating margin exactly 0.0: dropped. info carries no
      operating income to check it against, and an exact 0.0 is far likelier
      a placeholder than a real result.
    """
    notes: list[str] = []
    revenue = info.get("totalRevenue")
    if not revenue:
        dropped = [m for m in _MARGINS if out.pop(m, None) is not None]
        if dropped:
            notes.append(f"No revenue reported, so margins are undefined: dropped "
                         f"{', '.join(dropped)}.")
        return notes
    if out.get("profit_margin") == 0.0:
        net_income = info.get("netIncomeToCommon")
        if net_income:
            out["profit_margin"] = float(net_income) / float(revenue)
            notes.append(f"yfinance gave a net margin of 0.0; recomputed from net income "
                         f"{float(net_income) / 1e6:,.0f}M on revenue {float(revenue) / 1e6:,.0f}M "
                         f"as {out['profit_margin'] * 100:.1f}%.")
        else:
            del out["profit_margin"]
            notes.append("yfinance gave a net margin of 0.0 with no net income to check it; "
                         "treated as missing.")
    for name in ("gross_margin", "operating_margin"):
        if out.get(name) == 0.0:
            del out[name]
            notes.append(f"yfinance gave a {name.replace('_', ' ')} of exactly 0.0; treated as "
                         f"missing.")
    return notes


def _align_currencies(out: dict, info: dict) -> list[str]:
    """Put market-cap ratios in one currency, and the market cap in dollars.

    Yahoo divides its market cap (quote currency) by revenue (reporting
    currency) without converting, and likewise its price by book value:
    Sofwave's price/sales read 11.9 for a true 3.8. Checked on 2026-11-02 across
    Tel Aviv, London, Paris and US-listed foreign shares:

    - price/sales: wrong wherever the two currencies differ; recomputed.
    - price/book: wrong for local listings (BP 2.0, true 2.7); recomputed.
      For US-listed foreign shares Yahoo's book value is already in dollars
      (Alibaba, Novo), but TSMC's is on another share basis than its ADR, so
      these are left as given and flagged.
    - P/E: right as given. EPS is in the quote currency (BP 0.42 is its $0.58
      in pounds), so nothing is changed.
    """
    quote, reported = currency.quote_currency(info), currency.report_currency(info)
    out["_quote_currency"], out["_report_currency"] = quote, reported
    notes: list[str] = []
    divisor = currency.minor_divisor(info.get("currency"))
    if divisor != 1.0:
        notes.append(f"Prices are quoted in {info.get('currency')}, 1/{divisor:g} of a {quote}.")
    cap = out.get("market_cap")
    usd = currency.convert(cap, quote, "USD") if cap else None
    if usd is not None:
        out["market_cap_usd"] = usd
    if quote == reported:
        return notes

    cap_reported = currency.convert(cap, quote, reported) if cap else None
    if cap and cap_reported is None:
        notes.append(f"Trades in {quote} but reports in {reported}, and no exchange rate was "
                     f"available: price/sales and price/book mix the two.")
        return notes
    fixed = []
    revenue = out.get("total_revenue")
    if cap_reported and revenue and "price_to_sales" in out:
        was, out["price_to_sales"] = out["price_to_sales"], cap_reported / revenue
        fixed.append(f"price/sales {out['price_to_sales']:.2f} (Yahoo {was:.2f})")
    if cap_reported:
        out["_cap_reported"] = cap_reported
    if quote == "USD":
        if "price_to_book" in out:
            notes.append(f"Price/book is Yahoo's: for a US-listed foreign share its book value "
                         f"can be on a different share basis, so treat {out['price_to_book']:.2f} "
                         f"with care.")
    else:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        book = info.get("bookValue")
        price_reported = (currency.convert(float(price) / divisor, quote, reported)
                          if price else None)
        if price_reported and book and "price_to_book" in out:
            was, out["price_to_book"] = out["price_to_book"], price_reported / float(book)
            fixed.append(f"price/book {out['price_to_book']:.2f} (Yahoo {was:.2f})")
    if fixed:
        notes.append(f"Trades in {quote} but reports in {reported}: recomputed in one currency, "
                     f"{' and '.join(fixed)}.")
    return notes


def extract_fundamentals(info: dict, symbol: str) -> dict[str, float]:
    """The fields this module reads from a yfinance info dict, as floats.

    Fields yfinance did not provide are omitted rather than defaulted,
    placeholder margins are repaired (see _repair_margins) and market-cap ratios
    are put in one currency (see _align_currencies).
    """
    out: dict[str, float] = {}
    for raw_key, name in _RAW_FIELDS.items():
        value = info.get(raw_key)
        if value is None:
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue

    data_notes = _repair_margins(out, info)
    out["_currency_notes"] = " ".join(_align_currencies(out, info))

    # Sector/industry are strings, kept under private keys so they travel with
    # the metrics dict without polluting the numeric output.
    out["_sector"] = info.get("sector") or ""
    out["_industry"] = info.get("industry") or ""
    out["_name"] = info.get("shortName") or symbol.strip().upper()
    out["_data_notes"] = " ".join(data_notes)
    return out


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
    return extract_fundamentals(info, symbol)


def compute_ratios(fundamentals: dict[str, float]) -> dict[str, float]:
    """Derive additional ratios that yfinance does not supply directly."""
    derived: dict[str, float] = {}

    eps = fundamentals.get("trailing_eps")
    forward_eps = fundamentals.get("forward_eps")
    # Free cash flow and revenue are in the reporting currency; so must the cap be.
    market_cap = fundamentals.get("_cap_reported") or fundamentals.get("market_cap")
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


def max_peers() -> int:
    """How many peers one comparison fetches; each costs a rate-limited request."""
    return int(load_allowed_sources().get("max_peers", 6))


def load_custom_peers() -> dict[str, list[str]]:
    """The peers chosen per company in config/custom_peers.yaml; {} when there are none."""
    if not CUSTOM_PEERS_PATH.exists():
        return {}
    with CUSTOM_PEERS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError(f"{CUSTOM_PEERS_PATH} must list peers per company: `SYMBOL: [A, B]`.")
    return {str(k).upper(): [str(p).upper() for p in v] for k, v in data.items()}


def save_custom_peers(symbol: str, peers: list[str]) -> None:
    """Remember the peers for one company, keeping those of the others."""
    chosen = {**load_custom_peers(), symbol.strip().upper(): list(peers)}
    atomic_write_text(CUSTOM_PEERS_PATH, _CUSTOM_PEERS_HEADER
                      + yaml.safe_dump(chosen, default_flow_style=None))


def _resolve_peers(symbol: str, sector: str, industry: str) -> tuple[list[str], str, bool]:
    """Find the peer tickers for a symbol.

    Returns (peers, niche_label, is_sector_fallback). Prefers an industry-level
    group; falls back to the broader sector list when the niche is not declared.
    """
    config = load_allowed_sources()
    groups = config.get("peer_groups") or {}
    sector_defaults = config.get("default_peers_by_sector") or {}

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
    peers = [p for p in peers if p.upper() != symbol.upper()][:max_peers()]
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


def compare_to_peers(
    symbol: str,
    metrics: dict[str, float],
    ask_peers: Callable[[str, str, list[str]], list[str] | None] | None = None,
) -> PeerComparison:
    """Compare a symbol's valuation metrics against the median of its niche.

    Each peer costs one rate-limited request; peers that fail to fetch are
    skipped rather than aborting the comparison.

    Peers chosen for this company go first. With none, and no group for its
    industry, `ask_peers(symbol, niche, suggested)` may supply them; returning
    nothing keeps the sector list, if there is one.
    """
    symbol = symbol.strip().upper()
    sector = str(metrics.get("_sector", ""))
    industry = str(metrics.get("_industry", ""))
    peers, niche, fallback = _resolve_peers(symbol, sector, industry)

    custom = load_custom_peers().get(symbol)
    if not custom and fallback and ask_peers is not None:
        custom = ask_peers(symbol, niche, peers)
    if custom:
        peers = [p for p in custom if p != symbol][:max_peers()]
        niche, fallback = industry or niche, False
    is_custom = bool(custom)

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
                f"{sector!r}. Run research.py in a terminal to choose peers, or add "
                f"them to config/custom_peers.yaml."
                if industry or sector else
                "No industry or sector is listed for this symbol (an ETF or fund?), "
                "so there are no peers to compare with."
            ),
        )

    # trailing_eps is compared and charted but deliberately kept out of both
    # _LOWER_IS_CHEAPER and _HIGHER_IS_BETTER: EPS is not comparable across
    # companies as a valuation judgment (it scales with share count), so it
    # informs the reader without moving the verdict.
    comparable = list(_LOWER_IS_CHEAPER) + list(_HIGHER_IS_BETTER) + ["trailing_eps"]
    collected: dict[str, list[float]] = {name: [] for name in comparable}
    per_ticker: dict[str, dict[str, float]] = {name: {} for name in comparable}
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
                per_ticker[name][peer] = float(value)

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

    scope = ("chosen" if is_custom
             else "sector (no niche group declared)" if fallback else "niche")
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
        peer_values={k: v for k, v in per_ticker.items() if v},
        verdict=verdict,
        notes=notes,
        is_custom=is_custom,
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


def analyze(
    symbol: str,
    include_peers: bool = True,
    ask_peers: Callable[[str, str, list[str]], list[str] | None] | None = None,
) -> FundamentalRead:
    """Run the full FA pipeline for a symbol.

    Set include_peers=False to skip the peer comparison (faster: it avoids one
    rate-limited request per peer). `ask_peers` is explained at compare_to_peers.
    """
    symbol = symbol.strip().upper()
    raw = fetch_fundamentals(symbol)
    metrics = {**raw, **compute_ratios(raw)}

    peer_comparison = compare_to_peers(symbol, metrics, ask_peers) if include_peers else None
    signal, reasons = _classify(metrics, peer_comparison)
    if raw.get("_data_notes"):
        reasons.append(f"Data repaired: {raw['_data_notes']}")
    if raw.get("_currency_notes"):
        reasons.append(f"Currency: {raw['_currency_notes']}")

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
        name=str(raw.get("_name") or ""),
        sector=str(raw.get("_sector") or ""),
        quote_currency=str(raw.get("_quote_currency") or "USD"),
        report_currency=str(raw.get("_report_currency") or "USD"),
    )
