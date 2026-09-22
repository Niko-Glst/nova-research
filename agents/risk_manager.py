"""Correlation and risk-adjusted return checks.

Gates a candidate position before allocation_decision.py sizes it. Purely
analytical: no order placement anywhere in this module.

Two questions are asked, and they are independent:

1. **Does this position add anything?** A candidate that moves with the rest of
   the book concentrates existing risk rather than diversifying it. Correlation
   is measured against the portfolio's aggregate return series, not against each
   holding, because what matters is the effect on total portfolio variance.

2. **Is the return worth the volatility?** Sharpe divides excess return by total
   volatility; Sortino divides it by downside volatility only. Both are
   reported because they disagree in an informative way: a position with large
   upside swings scores badly on Sharpe and well on Sortino, and that gap is
   itself a description of the return shape.

Rejection is on a stated reason, never a bare boolean, so the gate can be
argued with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Gate thresholds. These are judgement calls, set here rather than inline so
# they can be changed in one place and seen at a glance.
MAX_CORRELATION = 0.85  # above this the position is a duplicate of the book
MIN_SHARPE = 0.0  # a negative Sharpe means volatility without compensation
MIN_OBSERVATIONS = 60  # below this the metrics are too noisy to gate on

# A standard deviation below this counts as zero. Floating point leaves a
# residue of roughly 1e-19 on a constant series, and `> 0` lets that through:
# dividing by it produced a Sharpe of 7.3e16 rather than the intended 0.
_ZERO_VOLATILITY = 1e-12

# Annual risk-free rate used for excess return. Set to zero by default: the
# figure moves, and a wrong constant is worse than an explicit simplification.
RISK_FREE_RATE = 0.0


@dataclass(frozen=True)
class RiskAssessment:
    symbol: str
    correlation_to_portfolio: float
    sharpe_ratio: float
    sortino_ratio: float
    approved: bool
    reason: str
    # Additional detail beyond the original six fields.
    annualized_return_pct: float = 0.0
    annualized_volatility_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    observations: int = 0
    diversification_benefit: float = 0.0  # 1 - |correlation|
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        lines = [
            f"Risk assessment: {self.symbol} [{verdict}]",
            f"  reason:              {self.reason}",
            f"  correlation to book: {self.correlation_to_portfolio:+.2f}",
            f"  Sharpe:              {self.sharpe_ratio:.2f}",
            f"  Sortino:             {self.sortino_ratio:.2f}",
            f"  annualized return:   {self.annualized_return_pct:+.1f}%",
            f"  annualized vol:      {self.annualized_volatility_pct:.1f}%",
            f"  max drawdown:        {self.max_drawdown_pct:.1f}%",
            f"  observations:        {self.observations}",
        ]
        if self.warnings:
            lines.append("  warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def compute_correlation(
    candidate_returns: pd.Series, portfolio_returns: pd.DataFrame
) -> float:
    """Correlation of a candidate's returns to the portfolio's aggregate return.

    Columns of `portfolio_returns` are individual holdings; they are equally
    weighted into one series first. Equal weighting is a stated simplification:
    with position sizes available the aggregate should be weighted by them, and
    that is what `portfolio_returns` should carry if the caller has it.

    Returns NaN when the overlap is too short or either series is constant,
    rather than a number that looks like a measurement.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        return float("nan")

    aggregate = portfolio_returns.mean(axis=1)
    aligned = pd.concat([candidate_returns, aggregate], axis=1, join="inner").dropna()

    if len(aligned) < 3:
        return float("nan")

    left, right = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if left.std() == 0 or right.std() == 0:
        return float("nan")

    return float(left.corr(right))


def compute_risk_adjusted_metrics(returns: pd.Series) -> dict[str, float]:
    """Sharpe, Sortino and the supporting figures for a daily return series.

    Both ratios are annualized by the square root of trading days, which
    assumes returns are serially independent. They are not, so treat the
    absolute values as comparative rather than exact.
    """
    clean = pd.Series(returns).dropna().astype(float)
    count = len(clean)

    if count < 2:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "observations": count,
        }

    daily_risk_free = RISK_FREE_RATE / TRADING_DAYS
    excess = clean - daily_risk_free

    mean = float(excess.mean())
    volatility = float(clean.std(ddof=1))

    downside = excess[excess < 0]
    downside_volatility = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    sharpe = (
        mean / volatility * np.sqrt(TRADING_DAYS)
        if volatility > _ZERO_VOLATILITY
        else 0.0
    )
    sortino = (
        mean / downside_volatility * np.sqrt(TRADING_DAYS)
        if downside_volatility > _ZERO_VOLATILITY
        else 0.0
    )

    equity = (1.0 + clean).cumprod()
    drawdown = (equity / equity.cummax() - 1.0).min() * 100.0
    years = count / TRADING_DAYS
    annualized = (
        (float(equity.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0
        if years > 0 and float(equity.iloc[-1]) > 0
        else 0.0
    )

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "annualized_return_pct": float(annualized),
        "annualized_volatility_pct": float(volatility * np.sqrt(TRADING_DAYS) * 100.0),
        "max_drawdown_pct": float(drawdown),
        "observations": count,
    }


def assess(
    symbol: str,
    candidate_returns: pd.Series,
    portfolio_returns: pd.DataFrame | None = None,
    max_correlation: float = MAX_CORRELATION,
    min_sharpe: float = MIN_SHARPE,
) -> RiskAssessment:
    """Gate a candidate position on correlation and risk-adjusted return.

    `portfolio_returns` may be None or empty for the first position in a book;
    the correlation check is then skipped rather than failed, and that is noted
    in the warnings.
    """
    symbol = symbol.strip().upper()
    metrics = compute_risk_adjusted_metrics(candidate_returns)
    correlation = compute_correlation(candidate_returns, portfolio_returns)

    warnings: list[str] = []
    if metrics["observations"] < MIN_OBSERVATIONS:
        warnings.append(
            f"Only {metrics['observations']} return observations; ratios computed "
            f"on fewer than {MIN_OBSERVATIONS} days are unstable."
        )
    if np.isnan(correlation):
        warnings.append(
            "No portfolio to correlate against, so the diversification check "
            "was skipped."
        )
    if metrics["sortino"] > 0 and metrics["sharpe"] > 0:
        ratio = metrics["sortino"] / metrics["sharpe"]
        if ratio > 1.8:
            warnings.append(
                f"Sortino is {ratio:.1f}x Sharpe: the volatility is mostly "
                f"upside, so Sharpe understates this position."
            )

    # --- the gate ---------------------------------------------------------
    approved = True
    reason = "Passes correlation and risk-adjusted return checks."

    if metrics["observations"] < 2:
        approved = False
        reason = "Insufficient return history to assess."
    elif metrics["sharpe"] < min_sharpe:
        approved = False
        reason = (
            f"Sharpe {metrics['sharpe']:.2f} is below the {min_sharpe:.2f} floor: "
            f"the position carries volatility without compensating return."
        )
    elif not np.isnan(correlation) and abs(correlation) > max_correlation:
        approved = False
        reason = (
            f"Correlation {correlation:+.2f} exceeds the {max_correlation:.2f} "
            f"limit: this duplicates risk already in the book rather than "
            f"diversifying it."
        )

    return RiskAssessment(
        symbol=symbol,
        correlation_to_portfolio=correlation,
        sharpe_ratio=metrics["sharpe"],
        sortino_ratio=metrics["sortino"],
        approved=approved,
        reason=reason,
        annualized_return_pct=metrics["annualized_return_pct"],
        annualized_volatility_pct=metrics["annualized_volatility_pct"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        observations=metrics["observations"],
        diversification_benefit=(
            0.0 if np.isnan(correlation) else 1.0 - abs(correlation)
        ),
        warnings=warnings,
    )
