"""SWOT analysis derived from measured figures.

Every entry names the number behind it. That is the constraint that keeps this
useful: a SWOT built from adjectives is a horoscope, while one where each line
carries the metric it rests on can be checked and argued with.

The four quadrants are split along the conventional axes:

- Strengths / Weaknesses: internal and present -- margins, growth, leverage,
  profitability as they stand today.
- Opportunities / Threats: external or forward-looking -- valuation relative to
  the peer group, insider behaviour, coverage trend, the price regime.

Where the data is missing a quadrant stays short rather than being padded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.fa_analyst import FundamentalRead
from agents.insider_analyst import InsiderRead
from agents.sentiment_analyst import SentimentRead
from agents.ta_analyst import TechnicalRead


@dataclass(frozen=True)
class SwotItem:
    """One finding, with the figure it rests on."""

    text: str
    metric: str  # the measurement behind the claim
    value: str  # that measurement, formatted


@dataclass(frozen=True)
class Swot:
    symbol: str
    strengths: list[SwotItem] = field(default_factory=list)
    weaknesses: list[SwotItem] = field(default_factory=list)
    opportunities: list[SwotItem] = field(default_factory=list)
    threats: list[SwotItem] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.strengths or self.weaknesses or self.opportunities or self.threats)


def _internal(metrics: dict[str, float]) -> tuple[list[SwotItem], list[SwotItem]]:
    """Split the company's own figures into strengths and weaknesses."""
    strengths: list[SwotItem] = []
    weaknesses: list[SwotItem] = []

    margin = metrics.get("profit_margin")
    if margin is not None:
        if margin > 0.20:
            strengths.append(
                SwotItem(
                    "Converts a high share of revenue into profit",
                    "Net margin",
                    f"{margin * 100:.1f}%",
                )
            )
        elif margin < 0:
            weaknesses.append(
                SwotItem(
                    "Loses money on every dollar of revenue",
                    "Net margin",
                    f"{margin * 100:.1f}%",
                )
            )

    operating = metrics.get("operating_margin")
    if operating is not None and operating > 0.25:
        strengths.append(
            SwotItem(
                "Operating leverage is strong before financing and tax",
                "Operating margin",
                f"{operating * 100:.1f}%",
            )
        )

    growth = metrics.get("revenue_growth")
    if growth is not None:
        if growth > 0.25:
            strengths.append(
                SwotItem(
                    "Revenue is compounding quickly",
                    "Revenue growth (YoY)",
                    f"{growth * 100:.1f}%",
                )
            )
        elif growth < 0:
            weaknesses.append(
                SwotItem(
                    "The top line is shrinking",
                    "Revenue growth (YoY)",
                    f"{growth * 100:.1f}%",
                )
            )

    roe = metrics.get("return_on_equity")
    if roe is not None and roe > 0.20:
        strengths.append(
            SwotItem(
                "Generates strong returns on shareholder capital",
                "Return on equity",
                f"{roe * 100:.1f}%",
            )
        )

    eps = metrics.get("trailing_eps")
    if eps is not None and eps <= 0:
        weaknesses.append(
            SwotItem(
                "Not yet profitable on a trailing basis",
                "Trailing EPS",
                f"{eps:.2f}",
            )
        )

    leverage = metrics.get("debt_to_equity")
    if leverage is not None and leverage > 150:
        weaknesses.append(
            SwotItem(
                "Carries heavy debt relative to equity",
                "Debt / equity",
                f"{leverage:.0f}%",
            )
        )

    current = metrics.get("current_ratio")
    if current is not None and current < 1.0:
        weaknesses.append(
            SwotItem(
                "Short-term liabilities exceed short-term assets",
                "Current ratio",
                f"{current:.2f}",
            )
        )

    fcf_yield = metrics.get("fcf_yield_pct")
    if fcf_yield is not None and fcf_yield > 4.0:
        strengths.append(
            SwotItem(
                "Throws off meaningful free cash flow against its market value",
                "FCF yield",
                f"{fcf_yield:.1f}%",
            )
        )

    return strengths, weaknesses


def _external(
    fundamental: FundamentalRead,
    technical: TechnicalRead | None,
    insider: InsiderRead | None,
    sentiment: SentimentRead | None,
) -> tuple[list[SwotItem], list[SwotItem]]:
    """Split the outside-facing figures into opportunities and threats."""
    opportunities: list[SwotItem] = []
    threats: list[SwotItem] = []

    # --- Valuation against the niche ---------------------------------------
    peers = fundamental.peer_comparison
    if peers is not None and peers.peers_used:
        forward_rel = peers.relative.get("forward_pe")
        growth_rel = peers.relative.get("revenue_growth")
        margin_rel = peers.relative.get("profit_margin")

        if forward_rel is not None and forward_rel < 0.85:
            opportunities.append(
                SwotItem(
                    f"Cheaper than {peers.niche} peers on forward earnings",
                    "Forward P/E vs peer median",
                    f"{forward_rel:.2f}x",
                )
            )
        elif forward_rel is not None and forward_rel > 1.3:
            threats.append(
                SwotItem(
                    f"Priced at a premium to {peers.niche} peers that results must justify",
                    "Forward P/E vs peer median",
                    f"{forward_rel:.2f}x",
                )
            )

        if growth_rel is not None and growth_rel > 1.5:
            opportunities.append(
                SwotItem(
                    "Growing materially faster than the peer group",
                    "Revenue growth vs peer median",
                    f"{growth_rel:.2f}x",
                )
            )

        if margin_rel is not None and margin_rel > 1.5:
            opportunities.append(
                SwotItem(
                    "More profitable than the typical peer, supporting a premium",
                    "Net margin vs peer median",
                    f"{margin_rel:.2f}x",
                )
            )
        elif margin_rel is not None and 0 < margin_rel < 0.7:
            threats.append(
                SwotItem(
                    "Less profitable than the typical peer",
                    "Net margin vs peer median",
                    f"{margin_rel:.2f}x",
                )
            )

    # --- Insider behaviour --------------------------------------------------
    if insider is not None and insider.signal != "unknown":
        if insider.buy_count > 0:
            opportunities.append(
                SwotItem(
                    f"Insiders are buying on the open market "
                    f"({insider.buy_count} purchases)",
                    "Net insider flow",
                    f"${insider.net_value:,.0f}",
                )
            )
        elif insider.senior_sell_count >= 3:
            threats.append(
                SwotItem(
                    f"Senior leadership is selling steadily "
                    f"({insider.senior_sell_count} C-suite sells)",
                    "Net insider flow",
                    f"${insider.net_value:,.0f}",
                )
            )

    # --- Price regime -------------------------------------------------------
    if technical is not None:
        close = technical.indicators.get("close")
        sma_slow = technical.indicators.get("sma_200")
        trailing_return = technical.indicators.get("return_pct")

        if close is not None and sma_slow is not None:
            if close < sma_slow:
                threats.append(
                    SwotItem(
                        "Trading below its long-term average: the trend is against it",
                        "Price vs 200-day average",
                        f"{close:.2f} vs {sma_slow:.2f}",
                    )
                )
            else:
                opportunities.append(
                    SwotItem(
                        "Holding above its long-term average",
                        "Price vs 200-day average",
                        f"{close:.2f} vs {sma_slow:.2f}",
                    )
                )

        if trailing_return is not None and trailing_return < -25:
            opportunities.append(
                SwotItem(
                    "Significant drawdown means much of the bad news may be priced in",
                    "Trailing period return",
                    f"{trailing_return:+.1f}%",
                )
            )

    # --- Attention ----------------------------------------------------------
    if sentiment is not None and sentiment.news_detail is not None:
        volume = sentiment.news_detail.volume
        if volume.status == "spike":
            threats.append(
                SwotItem(
                    "Coverage has surged: something is developing, direction unconfirmed",
                    "Coverage vs baseline",
                    f"{volume.ratio:.1f}x" if volume.ratio != float("inf") else "no baseline",
                )
            )
        elif volume.status == "quiet":
            opportunities.append(
                SwotItem(
                    "Attention has faded, which is where mispricings tend to sit",
                    "Coverage vs baseline",
                    f"{volume.ratio:.2f}x",
                )
            )

    return opportunities, threats


def build_swot(
    symbol: str,
    fundamental: FundamentalRead | None,
    technical: TechnicalRead | None,
    insider: InsiderRead | None,
    sentiment: SentimentRead | None,
) -> Swot:
    """Derive a SWOT from the measured figures across all analysts."""
    if fundamental is None:
        return Swot(symbol=symbol)

    strengths, weaknesses = _internal(fundamental.metrics)
    opportunities, threats = _external(fundamental, technical, insider, sentiment)

    return Swot(
        symbol=symbol,
        strengths=strengths,
        weaknesses=weaknesses,
        opportunities=opportunities,
        threats=threats,
    )
