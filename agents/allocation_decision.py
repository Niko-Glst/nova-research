"""Final position sizing, including transaction costs.

Takes conviction plus a risk assessment and produces a simulated position size.
This module never places a real order: it produces a research-grade sizing
recommendation for paper and simulation use only.

Three sizing methods, and the differences between them matter:

- **Fixed fractional** allocates a constant share of equity. Simple, and wrong
  in an obvious way: it puts the same money into a 15% volatility name and a
  70% one, so risk per position varies by a factor of four.
- **Volatility targeting** sizes so each position contributes the same expected
  volatility to the book. This is the default, because equalising risk
  contribution is what makes positions comparable.
- **Kelly** sizes by edge divided by variance. It maximises long-run growth
  *given correct probabilities*, and the probabilities here are estimates from
  a few years of history. Full Kelly on estimated inputs overbets badly, so the
  implementation applies a fractional multiplier (0.5 by default) and the
  rationale says so.

Every method is then clipped to a maximum position size and reduced for
correlation to the existing book, because sizing that ignores what you already
hold is not sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

TRADING_DAYS = 252

# The volatility each position is sized to contribute, annualized. 15% is a
# conventional single-position target; the book's total is lower when positions
# are not perfectly correlated.
DEFAULT_TARGET_VOLATILITY = 0.15

# No single position may exceed this share of equity, whatever the method says.
# A sizing model that recommends 60% of the book in one name is telling you its
# inputs are wrong, not that the position is that good.
MAX_POSITION_PCT = 20.0
MIN_POSITION_PCT = 0.5  # below this the position is not worth its costs

# Kelly is scaled down because its inputs are estimates. Half-Kelly gives up
# roughly a quarter of theoretical growth for a large reduction in variance,
# which is the right trade when the edge itself is uncertain.
KELLY_FRACTION = 0.5

# Round-trip cost estimate, in basis points, when the caller gives none.
DEFAULT_COST_BPS = 10.0


class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_TARGET = "volatility_target"
    KELLY = "kelly"


@dataclass(frozen=True)
class AllocationDecision:
    symbol: str
    position_size_pct: float  # percent of simulated portfolio equity
    estimated_transaction_cost: float
    rationale: str
    # Detail beyond the original four fields.
    method: str = SizingMethod.VOLATILITY_TARGET.value
    position_value: float = 0.0
    unconstrained_size_pct: float = 0.0  # before caps and correlation haircut
    cost_as_pct_of_position: float = 0.0
    expected_volatility_contribution_pct: float = 0.0
    constraints_applied: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.position_size_pct >= MIN_POSITION_PCT

    def summary(self) -> str:
        lines = [
            f"Allocation: {self.symbol}",
            f"  method:            {self.method}",
            f"  position size:     {self.position_size_pct:.2f}% of equity",
            f"  position value:    ${self.position_value:,.0f}",
            f"  unconstrained:     {self.unconstrained_size_pct:.2f}%",
            f"  transaction cost:  ${self.estimated_transaction_cost:,.2f} "
            f"({self.cost_as_pct_of_position:.3f}% of position)",
            f"  vol contribution:  {self.expected_volatility_contribution_pct:.2f}%",
            f"  rationale:         {self.rationale}",
        ]
        if self.constraints_applied:
            lines.append("  constraints:")
            lines += [f"    - {c}" for c in self.constraints_applied]
        return "\n".join(lines)


def _fixed_fractional_size(conviction: float, base_pct: float = 10.0) -> float:
    """A constant share of equity, scaled by conviction."""
    return base_pct * max(0.0, min(1.0, conviction))


def _volatility_target_size(
    conviction: float,
    annualized_volatility_pct: float,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
) -> float:
    """Size so the position contributes `target_volatility` to the book.

    A position of weight w in an asset with volatility v contributes w*v. Setting
    that equal to the target gives w = target / v, scaled by conviction.
    """
    volatility = annualized_volatility_pct / 100.0
    if volatility <= 0:
        return 0.0
    weight = target_volatility / volatility
    return weight * 100.0 * max(0.0, min(1.0, conviction))


def _kelly_size(
    expected_return_pct: float,
    annualized_volatility_pct: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """Kelly weight: edge over variance, scaled down for estimation error.

    For a continuous return the Kelly weight is mu / sigma^2. Both inputs are
    estimated from history, and the formula is unforgiving of an overstated mu,
    so the result is multiplied by `fraction`.
    """
    mu = expected_return_pct / 100.0
    sigma = annualized_volatility_pct / 100.0
    if sigma <= 0 or mu <= 0:
        return 0.0
    return (mu / (sigma**2)) * fraction * 100.0


def size_position(
    symbol: str,
    conviction: float,
    risk_assessment: dict,
    portfolio_equity: float,
    estimated_cost_bps: float = DEFAULT_COST_BPS,
    method: SizingMethod | str = SizingMethod.VOLATILITY_TARGET,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    max_position_pct: float = MAX_POSITION_PCT,
) -> AllocationDecision:
    """Compute a simulated position size. Never places an order.

    `risk_assessment` is a mapping carrying at least `annualized_volatility_pct`;
    `approved`, `correlation_to_portfolio` and `annualized_return_pct` are used
    when present. A RiskAssessment dataclass can be passed through
    `dataclasses.asdict`.
    """
    symbol = symbol.strip().upper()
    if portfolio_equity <= 0:
        raise ValueError("portfolio_equity must be positive")

    method = SizingMethod(method)
    conviction = max(0.0, min(1.0, float(conviction)))

    volatility = float(risk_assessment.get("annualized_volatility_pct") or 0.0)
    expected_return = float(risk_assessment.get("annualized_return_pct") or 0.0)
    correlation = risk_assessment.get("correlation_to_portfolio")
    approved = risk_assessment.get("approved", True)

    constraints: list[str] = []

    # A rejected candidate is sized at zero regardless of conviction: the gate
    # exists to be binding.
    if approved is False:
        return AllocationDecision(
            symbol=symbol,
            position_size_pct=0.0,
            estimated_transaction_cost=0.0,
            rationale=(
                "Risk assessment rejected this candidate, so no position is sized."
            ),
            method=method.value,
            constraints_applied=["Rejected by risk_manager"],
        )

    if method is SizingMethod.FIXED_FRACTIONAL:
        raw = _fixed_fractional_size(conviction)
        basis = f"fixed fractional at {conviction:.0%} conviction"
    elif method is SizingMethod.KELLY:
        raw = _kelly_size(expected_return, volatility) * conviction
        basis = (
            f"{KELLY_FRACTION:.0%} Kelly on a {expected_return:+.1f}% expected "
            f"return against {volatility:.1f}% volatility"
        )
    else:
        raw = _volatility_target_size(conviction, volatility, target_volatility)
        basis = (
            f"volatility targeting to {target_volatility:.0%} contribution "
            f"against {volatility:.1f}% realised volatility"
        )

    unconstrained = raw

    # Correlation haircut: a position that moves with the book adds less
    # diversification, so it earns less capital. At correlation 1.0 the size is
    # halved; at 0.0 it is untouched.
    if correlation is not None and correlation == correlation:  # not NaN
        haircut = 1.0 - 0.5 * abs(float(correlation))
        if haircut < 1.0:
            raw *= haircut
            constraints.append(
                f"Reduced {(1 - haircut) * 100:.0f}% for correlation "
                f"{float(correlation):+.2f} to the existing book"
            )

    if raw > max_position_pct:
        constraints.append(
            f"Capped at the {max_position_pct:.0f}% single-position limit "
            f"(model suggested {raw:.1f}%)"
        )
        raw = max_position_pct

    if 0 < raw < MIN_POSITION_PCT:
        constraints.append(
            f"Below the {MIN_POSITION_PCT:.1f}% minimum: the position would not "
            f"cover its own costs"
        )
        raw = 0.0

    position_value = portfolio_equity * raw / 100.0
    # Charged on both sides: entering and eventually exiting.
    cost = position_value * (estimated_cost_bps / 10_000.0) * 2

    return AllocationDecision(
        symbol=symbol,
        position_size_pct=round(raw, 3),
        estimated_transaction_cost=round(cost, 2),
        rationale=f"Sized by {basis}.",
        method=method.value,
        position_value=round(position_value, 2),
        unconstrained_size_pct=round(unconstrained, 3),
        cost_as_pct_of_position=(
            round(cost / position_value * 100.0, 4) if position_value > 0 else 0.0
        ),
        expected_volatility_contribution_pct=round(raw / 100.0 * volatility, 3),
        constraints_applied=constraints,
    )
