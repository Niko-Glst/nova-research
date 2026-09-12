"""Do the analyst signals actually predict returns?

research_report.py weights its four analysts 1.0 / 0.8 / 0.6 / 0.4. Those
numbers were chosen by argument -- fundamentals are measured, sentiment is a
lexicon read over a thin sample -- and never tested. This module tests them.

The method is deliberately simple: for a set of symbols, record each analyst's
stance and the return that followed, then regress the one on the other. If the
fundamental signal deserves twice the weight of sentiment, its coefficient
should be larger and more reliably non-zero.

**What this cannot tell you.** Read this before believing any output:

1. The stances are computed from *today's* data, so pairing them with *past*
   returns measures association, not prediction. Genuine validation needs
   point-in-time snapshots taken before the return period -- which this project
   does not store. `lookahead_warning` in the result says so on every run.
2. A few dozen symbols at one moment is a tiny sample. Coefficients will move a
   lot with small changes in the universe.
3. Survivorship: the symbols you think to test are the ones still listed.

Used as intended, this is a consistency check on the weighting, not a backtest
of the pipeline. A signal whose coefficient is indistinguishable from zero
across every reasonable universe probably deserves less weight; one that is
consistently strong deserves more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents import research_report, ta_analyst
from backtests.regression import CorrelationResult, RegressionResult, correlate, fit_ols

# The analysts whose stances are studied, in the order used for the design matrix.
SIGNAL_NAMES = ("fundamental", "technical", "insider", "sentiment")

# Stances mapped to the same -1/0/+1 scale the report uses.
_STANCE_VALUES = {
    "bullish": 1.0, "undervalued": 1.0,
    "neutral": 0.0, "fair": 0.0,
    "bearish": -1.0, "overvalued": -1.0,
}


@dataclass(frozen=True)
class SymbolObservation:
    """One symbol's signals paired with the return that followed."""

    symbol: str
    signals: dict[str, float]  # analyst -> -1/0/+1, NaN when unavailable
    combined_score: float
    forward_return_pct: float
    error: str = ""

    @property
    def is_usable(self) -> bool:
        return not self.error and not np.isnan(self.forward_return_pct)


@dataclass(frozen=True)
class SignalStudy:
    """Regression and correlation results across a universe of symbols."""

    observations: list[SymbolObservation]
    regression: RegressionResult | None
    combined_regression: RegressionResult | None
    correlations: list[CorrelationResult] = field(default_factory=list)
    return_window_days: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def lookahead_warning(self) -> str:
        return (
            "Signals are computed from current data and paired with PAST returns, "
            "so this measures association, not prediction. Point-in-time snapshots "
            "would be needed for a genuine predictive test."
        )

    @property
    def usable_count(self) -> int:
        return sum(1 for o in self.observations if o.is_usable)

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"  SIGNAL STUDY: {self.usable_count} symbols, "
            f"{self.return_window_days}-day return window",
            "=" * 68,
            "",
            f"  !! {self.lookahead_warning}",
            "",
        ]

        if self.correlations:
            lines += [
                "  Individual correlation with the return window:",
                f"    {'signal':<14} {'r':>8} {'p':>9} {'n':>5}",
                f"    {'-' * 14} {'-' * 8} {'-' * 9} {'-' * 5}",
            ]
            for c in self.correlations:
                mark = " *" if c.is_significant else ""
                lines.append(
                    f"    {c.name:<14} {c.correlation:>8.3f} {c.p_value:>9.4f} "
                    f"{c.n_observations:>5}{mark}"
                )
            lines.append("")

        if self.combined_regression is not None:
            lines += ["  Combined score as a single predictor:", ""]
            lines += ["  " + line for line in self.combined_regression.summary().split("\n")]
            lines.append("")

        if self.regression is not None:
            lines += ["  All four signals as separate predictors:", ""]
            lines += ["  " + line for line in self.regression.summary().split("\n")]
            lines.append("")

        if self.failures:
            lines += ["  Symbols that could not be analyzed:"]
            for symbol, reason in self.failures.items():
                lines.append(f"    {symbol}: {reason[:70]}")
            lines.append("")

        lines.append(
            "  Reading this: a coefficient whose p-value is above 0.05 is not "
            "distinguishable from zero at this sample size."
        )
        lines.append("=" * 68)
        return "\n".join(lines)


def _forward_return(symbol: str, window_days: int) -> float:
    """Return over the most recent `window_days` trading days, as a percentage."""
    period = "2y" if window_days > 180 else "1y"
    history = ta_analyst.fetch_price_history(symbol, period=period)
    close = history["Close"].astype(float)

    if len(close) <= window_days:
        raise ValueError(
            f"Only {len(close)} bars available for {symbol}, need more than "
            f"{window_days}."
        )
    start = float(close.iloc[-window_days - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        raise ValueError(f"Non-positive start price for {symbol}.")
    return (end / start - 1.0) * 100.0


def collect_observations(
    symbols: list[str],
    return_window_days: int = 63,
    include_peers: bool = False,
) -> tuple[list[SymbolObservation], dict[str, str]]:
    """Run the pipeline for each symbol and pair its stances with a realised return.

    `include_peers` defaults to False: a peer comparison costs one rate-limited
    request per peer, which makes a 30-symbol study take an hour.
    """
    observations: list[SymbolObservation] = []
    failures: dict[str, str] = {}

    for symbol in symbols:
        symbol = symbol.strip().upper()
        try:
            report = research_report.analyze(symbol, include_peers=include_peers)
            forward = _forward_return(symbol, return_window_days)
        except Exception as exc:
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            continue

        stances = {
            "fundamental": report.fundamental.signal if report.fundamental else None,
            "technical": report.technical.signal if report.technical else None,
            "insider": report.insider.signal if report.insider else None,
            "sentiment": report.sentiment.signal if report.sentiment else None,
        }
        signals = {
            name: _STANCE_VALUES.get((stance or "").lower(), float("nan"))
            for name, stance in stances.items()
        }

        observations.append(
            SymbolObservation(
                symbol=symbol,
                signals=signals,
                combined_score=report.score,
                forward_return_pct=forward,
            )
        )

    return observations, failures


def study_signals(
    symbols: list[str],
    return_window_days: int = 63,
    include_peers: bool = False,
) -> SignalStudy:
    """Regress analyst signals on realised returns across a universe of symbols."""
    observations, failures = collect_observations(
        symbols, return_window_days=return_window_days, include_peers=include_peers
    )
    usable = [o for o in observations if o.is_usable]

    correlations: list[CorrelationResult] = []
    regression: RegressionResult | None = None
    combined: RegressionResult | None = None

    if len(usable) >= 3:
        returns = np.array([o.forward_return_pct for o in usable])

        for name in SIGNAL_NAMES:
            values = np.array([o.signals.get(name, float("nan")) for o in usable])
            # A signal that is constant across the universe carries no
            # information here, and correlate() would divide by a zero variance.
            finite = values[~np.isnan(values)]
            if finite.size >= 3 and finite.std() > 0:
                correlations.append(correlate(values, returns, name=name))

        combined_scores = np.array([o.combined_score for o in usable])
        if combined_scores.std() > 0 and len(usable) >= 4:
            combined = fit_ols(
                combined_scores,
                returns,
                ["combined_score"],
                target_name=f"{return_window_days}d return %",
            )

        # The multi-predictor fit needs enough complete rows to be worth running.
        design_rows = [
            [o.signals.get(name, float("nan")) for name in SIGNAL_NAMES] for o in usable
        ]
        design = np.array(design_rows, dtype=float)
        complete = ~np.isnan(design).any(axis=1)
        varying = [
            index
            for index in range(design.shape[1])
            if complete.sum() > 0 and design[complete, index].std() > 0
        ]
        if complete.sum() >= len(varying) + 2 and varying:
            regression = fit_ols(
                design[np.ix_(complete, varying)],
                returns[complete],
                [SIGNAL_NAMES[i] for i in varying],
                target_name=f"{return_window_days}d return %",
            )

    return SignalStudy(
        observations=observations,
        regression=regression,
        combined_regression=combined,
        correlations=correlations,
        return_window_days=return_window_days,
        failures=failures,
    )


def observations_to_frame(observations: list[SymbolObservation]) -> pd.DataFrame:
    """Tabulate observations, for inspection or export."""
    rows = []
    for observation in observations:
        row = {"symbol": observation.symbol, **observation.signals}
        row["combined_score"] = observation.combined_score
        row["forward_return_pct"] = observation.forward_return_pct
        rows.append(row)
    return pd.DataFrame(rows)
