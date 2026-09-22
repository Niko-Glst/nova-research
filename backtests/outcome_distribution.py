"""Probability distribution of a stock's own future outcomes.

This answers a different question from backtest_engine.py. That module asks
whether a trading rule would have worked; this one asks what the stock itself
plausibly does over a horizon, expressed as probabilities rather than a single
projected number.

**Volatility regimes.** A single distribution over several years of history
averages together two different worlds: the calm stretches and the turbulent
ones. A stock that returns 12% a year with 18% volatility in quiet periods and
loses 30% with 45% volatility in stressed ones is badly described by the blend
of the two. Returns are therefore split by trailing realised volatility, and
the distribution is reported for the calm regime, the volatile regime, and the
unconditional blend. The gap between the calm and volatile columns is the
honest measure of how much the answer depends on which world you are in.

**What this is not.** These are the stock's own historical returns, resampled.
Nothing here conditions on the fundamental or technical read, because doing so
credibly needs point-in-time snapshots of those signals taken before each
return period, which this project does not store (see backtests/signal_study.py
for the same limitation). A company whose character has just changed -- a first
profitable year, a large acquisition -- is not well described by its own past,
and no amount of resampling fixes that. `caveats` states this on every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtests.monte_carlo import resample_returns

TRADING_DAYS = 252

# Window for the trailing volatility used to classify each day's regime.
# A month is long enough to be stable and short enough to react to a change.
REGIME_WINDOW = 21

# Days are split at this percentile of trailing volatility. Two-thirds calm and
# one-third volatile matches how markets actually spend their time better than
# an even split would.
REGIME_SPLIT_PERCENTILE = 67

# Drawdown thresholds reported as probabilities, in percent.
DRAWDOWN_THRESHOLDS = (10.0, 20.0, 30.0, 50.0)

# Return thresholds reported as probabilities, in percent.
RETURN_THRESHOLDS = (-50.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0, 100.0)

DEFAULT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

# Below this many observations a regime's distribution is not worth reporting.
MIN_REGIME_OBSERVATIONS = 60


@dataclass(frozen=True)
class RegimeOutcome:
    """The outcome distribution within one volatility regime."""

    name: str  # "calm" / "volatile" / "all"
    observations: int  # source days feeding the resample
    annualized_volatility_pct: float
    return_percentiles: dict[int, float]
    drawdown_percentiles: dict[int, float]
    probability_of_return_above: dict[float, float]
    probability_of_drawdown_beyond: dict[float, float]
    median_return_pct: float
    mean_return_pct: float
    expected_shortfall_pct: float  # mean of the worst 5% of outcomes

    @property
    def probability_of_loss(self) -> float:
        # P(return > 0) is stored; loss is its complement.
        return 1.0 - self.probability_of_return_above.get(0.0, 0.0)


@dataclass(frozen=True)
class OutcomeDistribution:
    """Forward outcome probabilities for one symbol, split by volatility regime."""

    symbol: str
    horizon_days: int
    num_simulations: int
    history_days: int
    regimes: dict[str, RegimeOutcome] = field(default_factory=dict)
    current_regime: str = "unknown"
    current_volatility_pct: float = 0.0
    # The volatility that separates the regimes FOR THIS SYMBOL. "Calm" is
    # relative: 46% annualized is calm for NVDA and extreme for a utility, so
    # the label is meaningless without this number beside it.
    regime_threshold_pct: float = 0.0
    caveats: list[str] = field(default_factory=list)

    @property
    def blended(self) -> RegimeOutcome | None:
        return self.regimes.get("all")

    @property
    def current(self) -> RegimeOutcome | None:
        """The distribution for the regime the stock is in right now."""
        return self.regimes.get(self.current_regime) or self.blended

    def summary(self) -> str:
        """A readable table of the probabilities, regime by regime."""
        blend = self.blended
        if blend is None:
            return f"{self.symbol}: not enough history to build a distribution."

        names = [n for n in ("calm", "volatile", "all") if n in self.regimes]
        width = 12

        lines = [
            "=" * 70,
            f"  OUTCOME PROBABILITIES: {self.symbol}",
            f"  {self.horizon_days}-day horizon, {self.num_simulations:,} paths, "
            f"{self.history_days} days of history",
            "=" * 70,
            "",
            f"  Currently in the {self.current_regime.upper()} regime for this "
            f"symbol: trailing volatility {self.current_volatility_pct:.1f}% "
            f"against a {self.regime_threshold_pct:.0f}% split.",
            f"  (Calm is relative. {self.current_volatility_pct:.0f}% would be "
            f"extreme for a utility and is ordinary for {self.symbol}.)",
            "",
            "  " + "regime".ljust(26) + "".join(n.rjust(width) for n in names),
            "  " + "-" * (26 + width * len(names)),
        ]

        def row(label: str, values: list[str]) -> str:
            return "  " + label.ljust(26) + "".join(v.rjust(width) for v in values)

        lines.append(
            row(
                "source days",
                [f"{self.regimes[n].observations}" for n in names],
            )
        )
        lines.append(
            row(
                "annualized volatility",
                [f"{self.regimes[n].annualized_volatility_pct:.1f}%" for n in names],
            )
        )
        lines.append("")
        lines.append("  RETURN DISTRIBUTION")
        for percentile in DEFAULT_PERCENTILES:
            lines.append(
                row(
                    f"  p{percentile}",
                    [
                        f"{self.regimes[n].return_percentiles[percentile]:+.1f}%"
                        for n in names
                    ],
                )
            )
        lines.append("")
        lines.append("  PROBABILITY OF RETURN ABOVE")
        for threshold in RETURN_THRESHOLDS:
            lines.append(
                row(
                    f"  {threshold:+.0f}%",
                    [
                        f"{self.regimes[n].probability_of_return_above[threshold]:.0%}"
                        for n in names
                    ],
                )
            )
        lines.append("")
        lines.append("  PROBABILITY OF DRAWDOWN BEYOND")
        for threshold in DRAWDOWN_THRESHOLDS:
            lines.append(
                row(
                    f"  -{threshold:.0f}%",
                    [
                        f"{self.regimes[n].probability_of_drawdown_beyond[threshold]:.0%}"
                        for n in names
                    ],
                )
            )
        lines.append("")
        lines.append("  TAIL")
        lines.append(
            row(
                "  median drawdown",
                [f"{self.regimes[n].drawdown_percentiles[50]:.1f}%" for n in names],
            )
        )
        lines.append(
            row(
                "  worst-5% mean return",
                [f"{self.regimes[n].expected_shortfall_pct:.1f}%" for n in names],
            )
        )

        if self.caveats:
            lines += ["", "  CAVEATS"]
            lines += [f"    - {c}" for c in self.caveats]

        lines.append("=" * 70)
        return "\n".join(lines)


def classify_regimes(
    returns: pd.Series,
    window: int = REGIME_WINDOW,
    split_percentile: float = REGIME_SPLIT_PERCENTILE,
) -> tuple[pd.Series, float]:
    """Label each day calm or volatile by its trailing realised volatility.

    Returns (labels, threshold). The threshold is the annualized volatility
    that separates the two regimes, reported so the caller can say what "calm"
    actually meant for this symbol rather than leaving it abstract.
    """
    trailing = returns.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100.0
    usable = trailing.dropna()

    if usable.empty:
        return pd.Series("unknown", index=returns.index), 0.0

    threshold = float(np.percentile(usable, split_percentile))
    labels = pd.Series("unknown", index=returns.index, dtype=object)
    labels[trailing <= threshold] = "calm"
    labels[trailing > threshold] = "volatile"
    return labels, threshold


def _summarize_paths(
    name: str,
    source: np.ndarray,
    paths: np.ndarray,
) -> RegimeOutcome:
    """Turn simulated paths into the percentile and probability tables."""
    equity = np.cumprod(1.0 + paths, axis=1)
    terminal = (equity[:, -1] - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = (equity / running_max - 1.0).min(axis=1) * 100.0

    worst_5 = terminal[terminal <= np.percentile(terminal, 5)]

    return RegimeOutcome(
        name=name,
        observations=int(source.size),
        annualized_volatility_pct=float(source.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0),
        return_percentiles={
            p: float(np.percentile(terminal, p)) for p in DEFAULT_PERCENTILES
        },
        drawdown_percentiles={
            p: float(np.percentile(drawdowns, p)) for p in DEFAULT_PERCENTILES
        },
        probability_of_return_above={
            t: float((terminal > t).mean()) for t in RETURN_THRESHOLDS
        },
        probability_of_drawdown_beyond={
            t: float((drawdowns <= -t).mean()) for t in DRAWDOWN_THRESHOLDS
        },
        median_return_pct=float(np.median(terminal)),
        mean_return_pct=float(terminal.mean()),
        expected_shortfall_pct=float(worst_5.mean()) if worst_5.size else float(terminal.min()),
    )


def build_distribution(
    symbol: str,
    price_history: pd.DataFrame,
    horizon_days: int = TRADING_DAYS,
    num_simulations: int = 5000,
    seed: int | None = 42,
) -> OutcomeDistribution:
    """Build the forward outcome distribution for a symbol, split by regime.

    `seed` defaults to a fixed value so the same input gives the same
    probabilities: a risk number that moves between runs is not usable.
    """
    if price_history is None or price_history.empty:
        raise ValueError("price_history is empty, so no distribution can be built")
    if "Close" not in price_history.columns:
        raise ValueError("price_history must contain a 'Close' column")

    close = price_history["Close"].astype(float)
    returns = close.pct_change().dropna()

    if returns.size < MIN_REGIME_OBSERVATIONS:
        raise ValueError(
            f"Only {returns.size} return observations for {symbol}; need at least "
            f"{MIN_REGIME_OBSERVATIONS} to estimate a distribution."
        )

    labels, threshold = classify_regimes(returns)
    caveats: list[str] = []

    regimes: dict[str, RegimeOutcome] = {}
    groups = {
        "calm": returns[labels == "calm"].to_numpy(),
        "volatile": returns[labels == "volatile"].to_numpy(),
        "all": returns.to_numpy(),
    }

    for name, source in groups.items():
        if source.size < MIN_REGIME_OBSERVATIONS:
            if name != "all":
                caveats.append(
                    f"The {name} regime had only {source.size} days, too few to "
                    f"model separately, so it is omitted."
                )
            continue
        paths = resample_returns(
            source,
            num_simulations=num_simulations,
            horizon_days=horizon_days,
            method="block",
            seed=seed,
        )
        regimes[name] = _summarize_paths(name, source, paths)

    # Which regime is the stock in today?
    current_regime = str(labels.iloc[-1]) if not labels.empty else "unknown"
    recent_volatility = float(
        returns.tail(REGIME_WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0
    )

    # Caveats that apply to every result, stated rather than buried.
    caveats.append(
        "These are the stock's own past returns resampled. They assume the "
        "future resembles the past, which is exactly what fails when a company "
        "changes character."
    )
    if horizon_days > returns.size / 2:
        caveats.append(
            f"The {horizon_days}-day horizon is long relative to the "
            f"{returns.size} days of history available, so the paths repeat the "
            f"same source data many times over."
        )
    if "calm" in regimes and "volatile" in regimes:
        calm_median = regimes["calm"].return_percentiles[50]
        volatile_median = regimes["volatile"].return_percentiles[50]
        caveats.append(
            f"Regime matters here: the median outcome is {calm_median:+.0f}% in "
            f"calm conditions against {volatile_median:+.0f}% in volatile ones "
            f"(split at {threshold:.0f}% annualized volatility)."
        )

    return OutcomeDistribution(
        symbol=symbol,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        history_days=int(returns.size),
        regimes=regimes,
        current_regime=current_regime,
        current_volatility_pct=recent_volatility,
        regime_threshold_pct=threshold,
        caveats=caveats,
    )


def historical_drawdowns(price_history: pd.DataFrame) -> dict[str, float]:
    """How often this stock actually fell by each threshold, from real prices.

    The simulated probabilities answer "how often would this happen"; this
    answers "how often did it". When the two disagree sharply, the resampling
    assumption is the thing to doubt.
    """
    close = price_history["Close"].astype(float)
    running_max = close.cummax()
    drawdown = (close / running_max - 1.0) * 100.0

    return {
        "worst_drawdown_pct": float(drawdown.min()),
        "median_drawdown_pct": float(drawdown.median()),
        **{
            f"days_beyond_{int(t)}pct": float((drawdown <= -t).mean())
            for t in DRAWDOWN_THRESHOLDS
        },
    }
