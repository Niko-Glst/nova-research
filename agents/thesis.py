"""Investment thesis: what must be true, and what would break it.

This module deliberately does NOT emit a buy or sell instruction. It states the
conditions under which the numbers would support a positive view, with the
current reading of each condition next to it, so the reader can see how far the
company is from clearing each bar and decide for themselves. SECURITY.md keeps
this project on the research side of that line.

The conditions are derived from the measured data, not from a model's opinion:
each one names a metric, a threshold, and where the company currently stands.
That makes every line falsifiable -- you can check it against the company's next
filing -- which is the point of writing a thesis down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.fa_analyst import FundamentalRead
from agents.insider_analyst import InsiderRead
from agents.sentiment_analyst import SentimentRead
from agents.ta_analyst import TechnicalRead


@dataclass(frozen=True)
class Condition:
    """One falsifiable statement about what the numbers would need to show."""

    claim: str  # what must be true
    current: str  # where it stands today
    met: bool | None  # True/False, or None when the data is missing
    metric: str = ""  # which figure this rests on

    @property
    def status(self) -> str:
        if self.met is None:
            return "unknown"
        return "met" if self.met else "unmet"


@dataclass(frozen=True)
class Thesis:
    """The bull case, its current standing, and what would invalidate it."""

    symbol: str
    stance: str  # "constructive" / "mixed" / "skeptical"
    summary: str
    bull_case: list[Condition] = field(default_factory=list)
    breaks_if: list[str] = field(default_factory=list)
    conditions_met: int = 0
    conditions_total: int = 0

    @property
    def met_fraction(self) -> float:
        return self.conditions_met / self.conditions_total if self.conditions_total else 0.0


def _profitability_conditions(metrics: dict[str, float]) -> list[Condition]:
    """Conditions about earning money, or credibly getting there."""
    conditions: list[Condition] = []

    eps = metrics.get("trailing_eps")
    forward_eps = metrics.get("forward_eps")

    if eps is not None and eps <= 0:
        # Loss-making: the thesis depends entirely on the turn.
        if forward_eps is not None and forward_eps > 0:
            conditions.append(
                Condition(
                    claim="The forecast turn to profitability actually lands",
                    current=(
                        f"Trailing EPS {eps:.2f} (loss); forward estimate "
                        f"{forward_eps:.2f}. The market is paying for a turn that "
                        f"has not happened yet."
                    ),
                    met=False,
                    metric="trailing_eps",
                )
            )
        else:
            conditions.append(
                Condition(
                    claim="The company reaches profitability at all",
                    current=f"Trailing EPS {eps:.2f}, with no positive forward estimate.",
                    met=False,
                    metric="trailing_eps",
                )
            )
    elif eps is not None:
        conditions.append(
            Condition(
                claim="Earnings stay positive",
                current=f"Trailing EPS {eps:.2f}.",
                met=True,
                metric="trailing_eps",
            )
        )

    margin = metrics.get("profit_margin")
    if margin is not None:
        conditions.append(
            Condition(
                claim="Net margin holds above 10%",
                current=f"Net margin {margin * 100:.1f}%.",
                met=margin > 0.10,
                metric="profit_margin",
            )
        )

    return conditions


def _growth_conditions(metrics: dict[str, float]) -> list[Condition]:
    """Conditions about growth being real and durable."""
    conditions: list[Condition] = []

    growth = metrics.get("revenue_growth")
    if growth is not None:
        conditions.append(
            Condition(
                claim="Revenue growth stays above 15%",
                current=f"Revenue growing {growth * 100:.1f}% year over year.",
                met=growth > 0.15,
                metric="revenue_growth",
            )
        )

    implied = metrics.get("eps_growth_implied")
    if implied is not None and implied > 50:
        # A large implied jump is the load-bearing assumption in the price.
        conditions.append(
            Condition(
                claim=f"The implied {implied:+.0f}% EPS jump is delivered, not just forecast",
                current=(
                    "Forward estimates carry most of the valuation. If the next "
                    "few quarters miss, the multiple has nothing to stand on."
                ),
                met=None,
                metric="eps_growth_implied",
            )
        )

    return conditions


def _valuation_conditions(fundamental: FundamentalRead) -> list[Condition]:
    """Conditions about the price paid relative to the niche."""
    conditions: list[Condition] = []
    peers = fundamental.peer_comparison

    if peers is None or not peers.peers_used:
        return conditions

    forward_pe_rel = peers.relative.get("forward_pe")
    trailing_pe_rel = peers.relative.get("trailing_pe")

    if forward_pe_rel is not None:
        if forward_pe_rel > 1.5:
            conditions.append(
                Condition(
                    claim=(
                        f"The premium to {peers.niche} peers is earned by "
                        f"execution, not just expectation"
                    ),
                    current=(
                        f"Trades at {forward_pe_rel:.1f}x the peer median forward "
                        f"P/E. That gap has to be closed by results."
                    ),
                    met=False,
                    metric="forward_pe",
                )
            )
        else:
            conditions.append(
                Condition(
                    claim=f"Valuation stays reasonable against {peers.niche} peers",
                    current=(
                        f"Forward P/E at {forward_pe_rel:.2f}x the peer median "
                        f"-- at or below the group."
                    ),
                    met=True,
                    metric="forward_pe",
                )
            )
    elif trailing_pe_rel is not None:
        conditions.append(
            Condition(
                claim=f"Valuation stays reasonable against {peers.niche} peers",
                current=f"Trailing P/E at {trailing_pe_rel:.2f}x the peer median.",
                met=trailing_pe_rel <= 1.2,
                metric="trailing_pe",
            )
        )

    # Growth versus the group: a premium is defensible if growth is too.
    growth_rel = peers.relative.get("revenue_growth")
    if growth_rel is not None and growth_rel > 1.0:
        conditions.append(
            Condition(
                claim="Growth stays ahead of the peer group",
                current=(
                    f"Revenue growing {growth_rel:.1f}x the peer median rate. "
                    f"This is what would justify paying up."
                ),
                met=True,
                metric="revenue_growth",
            )
        )

    return conditions


def _trend_conditions(technical: TechnicalRead | None) -> list[Condition]:
    """Conditions about price confirming, rather than contradicting, the story."""
    if technical is None:
        return []

    indicators = technical.indicators
    close = indicators.get("close")
    sma_slow = indicators.get("sma_200")

    if close is None or sma_slow is None:
        return []

    above = close > sma_slow
    return [
        Condition(
            claim="Price reclaims and holds its 200-day average",
            current=(
                f"Price {close:.2f} is "
                f"{'above' if above else 'below'} the 200-day average "
                f"({sma_slow:.2f})."
            ),
            met=above,
            metric="sma_200",
        )
    ]


def _insider_conditions(insider: InsiderRead | None) -> list[Condition]:
    """Conditions about management putting their own money in."""
    if insider is None or insider.signal == "unknown":
        return [
            Condition(
                claim="Insiders buy on the open market",
                current="No insider filings available for this window.",
                met=None,
                metric="insider",
            )
        ]

    if insider.buy_count > 0:
        return [
            Condition(
                claim="Insider buying continues",
                current=(
                    f"{insider.buy_count} open-market buys "
                    f"(${insider.buy_value:,.0f}) in the last "
                    f"{insider.lookback_days} days."
                ),
                met=True,
                metric="insider",
            )
        ]

    return [
        Condition(
            claim="Insider selling slows, or turns into buying",
            current=(
                f"{insider.sell_count} sells (${insider.sell_value:,.0f}) and no "
                f"open-market buys. Routine at large caps, but it is not a vote "
                f"of confidence either."
            ),
            met=False,
            metric="insider",
        )
    ]


def _breaks_if(
    metrics: dict[str, float],
    fundamental: FundamentalRead,
    technical: TechnicalRead | None,
    insider: InsiderRead | None,
    sentiment: SentimentRead | None,
) -> list[str]:
    """What would invalidate the constructive case."""
    breaks: list[str] = []

    growth = metrics.get("revenue_growth")
    if growth is not None and growth > 0.15:
        breaks.append(
            f"Revenue growth decelerates below 15% (now {growth * 100:.0f}%). "
            f"The valuation rests on this continuing."
        )

    margin = metrics.get("profit_margin")
    if margin is not None and margin > 0:
        breaks.append(
            f"Net margin turns negative (now {margin * 100:.1f}%), showing the "
            f"growth is being bought rather than earned."
        )
    elif margin is not None:
        breaks.append(
            f"Losses deepen from {margin * 100:.1f}% instead of narrowing, "
            f"pushing profitability further out."
        )

    peers = fundamental.peer_comparison
    if peers is not None and peers.verdict == "expensive":
        breaks.append(
            f"Peers re-rate upward while this name stalls, leaving the premium "
            f"to {peers.niche} unjustified."
        )

    leverage = metrics.get("debt_to_equity")
    if leverage is not None and leverage > 100:
        breaks.append(
            f"Refinancing gets harder: debt/equity is already {leverage:.0f}%."
        )

    if technical is not None:
        sma_slow = technical.indicators.get("sma_200")
        close = technical.indicators.get("close")
        if close is not None and sma_slow is not None and close > sma_slow:
            breaks.append(
                f"Price loses the 200-day average ({sma_slow:.2f}), turning the "
                f"trend against the thesis."
            )

    if insider is not None and insider.senior_sell_count > 0:
        breaks.append(
            f"Senior leadership keeps selling ({insider.senior_sell_count} sells "
            f"by C-suite in the window) while the story is supposedly improving."
        )

    if sentiment is not None and sentiment.news_detail is not None:
        volume = sentiment.news_detail.volume
        if volume.status == "spike":
            breaks.append(
                f"The current {volume.ratio:.1f}x burst in coverage turns out to "
                f"be bad news rather than good -- volume says how loudly, not how well."
            )

    return breaks


def build_thesis(
    symbol: str,
    fundamental: FundamentalRead | None,
    technical: TechnicalRead | None,
    insider: InsiderRead | None,
    sentiment: SentimentRead | None,
) -> Thesis:
    """Assemble the conditions that would have to hold for a constructive view."""
    if fundamental is None:
        return Thesis(
            symbol=symbol,
            stance="mixed",
            summary=(
                f"No fundamental data for {symbol}, so no thesis can be stated. "
                f"Everything below rests on price and flow alone."
            ),
        )

    metrics = fundamental.metrics
    conditions: list[Condition] = []
    conditions += _profitability_conditions(metrics)
    conditions += _growth_conditions(metrics)
    conditions += _valuation_conditions(fundamental)
    conditions += _trend_conditions(technical)
    conditions += _insider_conditions(insider)

    decided = [c for c in conditions if c.met is not None]
    met = sum(1 for c in decided if c.met)
    total = len(decided)

    fraction = met / total if total else 0.0
    if fraction >= 0.6:
        stance = "constructive"
    elif fraction >= 0.35:
        stance = "mixed"
    else:
        stance = "skeptical"

    unmet = [c for c in decided if not c.met]
    summary = (
        f"{met} of {total} conditions for a constructive case currently hold. "
    )
    if unmet:
        summary += (
            f"The case rests on {len(unmet)} thing(s) changing: "
            + "; ".join(c.claim.lower() for c in unmet[:3])
            + "."
        )
    else:
        summary += "Every measurable condition is currently met."

    return Thesis(
        symbol=symbol,
        stance=stance,
        summary=summary,
        bull_case=conditions,
        breaks_if=_breaks_if(metrics, fundamental, technical, insider, sentiment),
        conditions_met=met,
        conditions_total=total,
    )
