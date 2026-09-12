"""Single-symbol research orchestrator.

Runs the four available analysts for one ticker -- technical, fundamental (with
niche peer comparison), sentiment, and insider activity -- and combines their
stances into one weighted verdict with the reasoning behind it.

The weights encode how much each leg deserves to be trusted:

- Fundamentals (1.0) carry the most, because they are measured rather than
  inferred, and the peer comparison makes valuation a relative judgment.
- Technicals (0.8) describe the current trend, which is real but shorter-lived.
- Insider activity (0.6) is a genuine signal but a noisy one -- routine selling
  dominates the data at large caps (see agents/insider_analyst.py).
- Sentiment (0.4) is the weakest, being a lexicon read over a thin sample.

A leg that could not be computed contributes nothing rather than counting as
neutral: absence of data is not evidence of balance.

This produces a research verdict, not advice, and places no orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import fa_analyst, insider_analyst, sentiment_analyst, strategy_agent
from agents import swot as swot_module
from agents import ta_analyst, thesis as thesis_module
from backtests import backtest_engine, monte_carlo

# How much each leg counts toward the combined score.
WEIGHTS = {
    "fundamental": 1.0,
    "technical": 0.8,
    "insider": 0.6,
    "sentiment": 0.4,
}

# Stance vocabulary differs per analyst; this maps each to a numeric direction.
_STANCE_VALUES = {
    "bullish": 1.0,
    "undervalued": 1.0,
    "neutral": 0.0,
    "fair": 0.0,
    "bearish": -1.0,
    "overvalued": -1.0,
}

# Score thresholds for the final verdict, on a -1..1 scale.
_POSITIVE_THRESHOLD = 0.25
_NEGATIVE_THRESHOLD = -0.25


@dataclass(frozen=True)
class ResearchReport:
    """The combined read across every analyst that produced a result."""

    symbol: str
    verdict: str  # "positive" / "negative" / "mixed" / "inconclusive"
    score: float  # -1.0 .. 1.0
    confidence: float  # 0.0 .. 1.0, share of total weight that reported
    technical: ta_analyst.TechnicalRead | None = None
    fundamental: fa_analyst.FundamentalRead | None = None
    sentiment: sentiment_analyst.SentimentRead | None = None
    insider: insider_analyst.InsiderRead | None = None
    errors: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    thesis: thesis_module.Thesis | None = None
    swot: swot_module.Swot | None = None
    backtest: backtest_engine.BacktestResult | None = None
    monte_carlo: monte_carlo.MonteCarloResult | None = None
    # Terminal returns from the simulation, kept for the histogram. Excluded
    # from repr: this is thousands of floats.
    simulated_returns: object = field(default=None, repr=False)
    buy_hold_equity: object = field(default=None, repr=False)


def _stance_value(stance: str | None) -> float | None:
    """Map an analyst stance to -1/0/+1, or None when it carries no direction."""
    if stance is None:
        return None
    return _STANCE_VALUES.get(stance.lower())


def analyze(
    symbol: str,
    period: str = "1y",
    include_peers: bool = True,
    insider_lookback_days: int = insider_analyst.DEFAULT_LOOKBACK_DAYS,
    strict_allowlist: bool = True,
    run_simulation: bool = False,
    backtest_period: str = "5y",
    num_simulations: int = 2000,
) -> ResearchReport:
    """Run every analyst for a symbol and combine their verdicts.

    A failing analyst is recorded in `errors` and excluded from the score; one
    broken data source should not cost you the other three reads.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")

    errors: dict[str, str] = {}

    technical: ta_analyst.TechnicalRead | None = None
    try:
        technical = ta_analyst.analyze(symbol, period=period)
    except Exception as exc:
        errors["technical"] = f"{type(exc).__name__}: {exc}"

    fundamental: fa_analyst.FundamentalRead | None = None
    try:
        fundamental = fa_analyst.analyze(symbol, include_peers=include_peers)
    except Exception as exc:
        errors["fundamental"] = f"{type(exc).__name__}: {exc}"

    sentiment: sentiment_analyst.SentimentRead | None = None
    try:
        sentiment = sentiment_analyst.analyze(symbol, strict_allowlist=strict_allowlist)
    except Exception as exc:
        errors["sentiment"] = f"{type(exc).__name__}: {exc}"

    insider: insider_analyst.InsiderRead | None = None
    try:
        insider = insider_analyst.analyze(symbol, lookback_days=insider_lookback_days)
    except Exception as exc:
        errors["insider"] = f"{type(exc).__name__}: {exc}"

    # --- Backtest and simulation (opt-in: it refetches a longer history) ---
    backtest_result = None
    simulation = None
    simulated = None
    buy_hold = None
    if run_simulation:
        try:
            history = ta_analyst.fetch_price_history(symbol, period=backtest_period)
            signals = strategy_agent.moving_average_crossover(history, symbol=symbol)
            backtest_result = backtest_engine.run_backtest(
                history, signals.entries, signals.exits, fees_bps=10, symbol=symbol
            )

            import numpy as _np

            equity = backtest_result.equity_curve.to_numpy()
            closes = history["Close"].astype(float).to_numpy()
            buy_hold = closes / closes[0]

            strategy_returns = _np.diff(equity) / equity[:-1]
            if strategy_returns.size > 2 and strategy_returns.std() > 0:
                simulation = monte_carlo.run_monte_carlo(
                    symbol,
                    strategy_returns,
                    num_simulations=num_simulations,
                    observed_return_pct=backtest_result.annualized_return_pct,
                    seed=42,
                )
                paths = monte_carlo.resample_returns(
                    strategy_returns, num_simulations, 252, seed=42
                )
                simulated = (_np.cumprod(1.0 + paths, axis=1)[:, -1] - 1.0) * 100.0
        except Exception as exc:
            errors["backtest"] = f"{type(exc).__name__}: {exc}"

    # --- Combine ----------------------------------------------------------
    legs = {
        "technical": technical.signal if technical else None,
        "fundamental": fundamental.signal if fundamental else None,
        "sentiment": sentiment.signal if sentiment else None,
        "insider": insider.signal if insider else None,
    }

    weighted_sum = 0.0
    reporting_weight = 0.0

    for leg, stance in legs.items():
        value = _stance_value(stance)
        if value is None:  # missing, or an "unknown" stance
            continue
        weight = WEIGHTS[leg]
        weighted_sum += value * weight
        reporting_weight += weight

    total_weight = sum(WEIGHTS.values())
    confidence = reporting_weight / total_weight if total_weight else 0.0
    score = weighted_sum / reporting_weight if reporting_weight else 0.0

    if reporting_weight == 0:
        verdict = "inconclusive"
    elif score >= _POSITIVE_THRESHOLD:
        verdict = "positive"
    elif score <= _NEGATIVE_THRESHOLD:
        verdict = "negative"
    else:
        verdict = "mixed"

    reported = [leg for leg, stance in legs.items() if _stance_value(stance) is not None]
    summary = (
        f"{symbol}: {verdict} (score {score:+.2f}, confidence {confidence:.0%} "
        f"from {len(reported)} of {len(WEIGHTS)} analysts)."
    )

    return ResearchReport(
        symbol=symbol,
        verdict=verdict,
        score=score,
        confidence=confidence,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        insider=insider,
        errors=errors,
        summary=summary,
        thesis=thesis_module.build_thesis(
            symbol, fundamental, technical, insider, sentiment
        ),
        swot=swot_module.build_swot(
            symbol, fundamental, technical, insider, sentiment
        ),
        backtest=backtest_result,
        monte_carlo=simulation,
        simulated_returns=simulated,
        buy_hold_equity=buy_hold,
    )


def _format_money(value: float) -> str:
    """Render a currency amount compactly (1.2B rather than 1200000000)."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.2f}"


def format_report(report: ResearchReport) -> str:
    """Render a report as readable plain text for terminal output."""
    width = 72
    lines: list[str] = [
        "=" * width,
        f"  RESEARCH REPORT: {report.symbol}",
        "=" * width,
        "",
        f"  VERDICT: {report.verdict.upper()}",
        f"  Score:      {report.score:+.2f}  (-1 bearish .. +1 bullish)",
        f"  Confidence: {report.confidence:.0%}  (share of analysts reporting)",
        "",
    ]

    # --- Fundamentals -----------------------------------------------------
    if report.fundamental is not None:
        fundamental = report.fundamental
        lines += ["-" * width, f"  FUNDAMENTALS -> {fundamental.signal.upper()}", "-" * width]

        metrics = fundamental.metrics
        eps = metrics.get("trailing_eps")
        forward_eps = metrics.get("forward_eps")
        pe = metrics.get("trailing_pe")
        market_cap = metrics.get("market_cap")

        if eps is not None:
            row = f"  EPS (trailing):   {eps:>10.2f}"
            if forward_eps is not None:
                row += f"   forward: {forward_eps:.2f}"
            lines.append(row)
        if pe is not None:
            lines.append(f"  P/E (trailing):   {pe:>10.2f}")
        if metrics.get("earnings_yield_pct") is not None:
            lines.append(
                f"  Earnings yield:   {metrics['earnings_yield_pct']:>9.2f}%"
            )
        if market_cap is not None:
            lines.append(f"  Market cap:       {_format_money(market_cap):>10}")
        if metrics.get("profit_margin") is not None:
            lines.append(f"  Net margin:       {metrics['profit_margin'] * 100:>9.1f}%")
        if metrics.get("revenue_growth") is not None:
            lines.append(f"  Revenue growth:   {metrics['revenue_growth'] * 100:>9.1f}%")

        lines.append("")
        for reason in fundamental.reasons:
            lines.append(f"    - {reason}")

        # --- Peer comparison ---------------------------------------------
        peers = fundamental.peer_comparison
        if peers is not None and peers.peers_used:
            scope = "SECTOR" if peers.is_sector_fallback else "NICHE"
            lines += [
                "",
                f"  {scope} COMPARISON: {peers.niche} -> {peers.verdict.upper()}",
                f"  Peers: {', '.join(peers.peers_used)}",
                "",
                f"    {'metric':<20} {'this':>10} {'peer med':>10} {'vs peers':>10}",
                f"    {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}",
            ]
            for name in sorted(peers.relative):
                own = peers.symbol_metrics.get(name)
                med = peers.peer_medians.get(name)
                rel = peers.relative[name]
                if own is None or med is None:
                    continue
                lines.append(
                    f"    {name:<20} {own:>10.2f} {med:>10.2f} {rel:>9.2f}x"
                )
        elif peers is not None:
            lines += ["", f"  PEER COMPARISON: {peers.notes}"]
        lines.append("")

    # --- Technicals -------------------------------------------------------
    if report.technical is not None:
        technical = report.technical
        lines += ["-" * width, f"  TECHNICALS -> {technical.signal.upper()}", "-" * width]
        for name, value in technical.indicators.items():
            lines.append(f"  {name:<18} {value:>12.2f}")
        lines.append("")
        for reason in technical.reasons:
            lines.append(f"    - {reason}")
        lines.append("")

    # --- Insider ----------------------------------------------------------
    if report.insider is not None:
        insider = report.insider
        lines += ["-" * width, f"  INSIDER ACTIVITY -> {insider.signal.upper()}", "-" * width]
        lines += [
            f"  Window:           last {insider.lookback_days} days",
            f"  Open-market buys: {insider.buy_count:>4}   "
            f"{_format_money(insider.buy_value)}",
            f"  Open-market sells:{insider.sell_count:>4}   "
            f"{_format_money(insider.sell_value)}",
            f"  Net flow:         {_format_money(insider.net_value)}",
            "",
        ]
        for reason in insider.reasons:
            lines.append(f"    - {reason}")
        lines.append("")

    # --- Sentiment --------------------------------------------------------
    if report.sentiment is not None:
        sentiment = report.sentiment
        lines += ["-" * width, f"  SENTIMENT -> {sentiment.signal.upper()}", "-" * width]
        provider = f" via {sentiment.news_provider}" if sentiment.news_provider else ""
        lines += [
            f"  News score:       {sentiment.news_score:>+8.2f} "
            f"({sentiment.news_sample} articles{provider})",
            f"  Reddit score:     {sentiment.reddit_score:>+8.2f} "
            f"({sentiment.reddit_sample} posts)",
        ]

        detail = sentiment.news_detail
        if detail is not None and detail.article_count:
            volume = detail.volume
            lines += [
                f"  Unweighted:       {detail.unweighted_score:>+8.2f} "
                f"(before recency weighting)",
                f"  Effective sample: {detail.effective_sample:>8.1f} "
                f"(weight-adjusted article count)",
                "",
                f"  COVERAGE VOLUME -> {volume.status.upper()}",
            ]
            if volume.status != "unknown":
                ratio = "inf" if volume.ratio == float("inf") else f"{volume.ratio:.2f}x"
                lines += [
                    f"    last {volume.recent_window_days}d:   "
                    f"{volume.recent_count:>3} articles "
                    f"({volume.recent_per_day:.2f}/day)",
                    f"    baseline:  {volume.baseline_count:>3} articles "
                    f"({volume.baseline_per_day:.2f}/day)",
                    f"    ratio:     {ratio}",
                ]
            if detail.top_positive:
                lines += ["", "    Most positive:"]
                for headline in detail.top_positive:
                    lines.append(f"      + {headline.title[:62]}")
                    if headline.url:
                        lines.append(f"        {headline.url}")
            if detail.top_negative:
                lines += ["", "    Most negative:"]
                for headline in detail.top_negative:
                    lines.append(f"      - {headline.title[:62]}")
                    if headline.url:
                        lines.append(f"        {headline.url}")

        lines.append("")
        for reason in sentiment.reasons:
            lines.append(f"    - {reason}")
        if sentiment.sources_unavailable:
            lines += ["", "    Sources not used:"]
            for entry in sentiment.sources_unavailable:
                name, _, detail = entry.partition(" (")
                reason = detail.rstrip(")").split(".")[0] if detail else "unavailable"
                lines.append(f"      {name}: {reason}")
        lines.append("")

    # --- Thesis -----------------------------------------------------------
    if report.thesis is not None and report.thesis.bull_case:
        thesis = report.thesis
        lines += ["-" * width, "  WHAT MUST BE TRUE", "-" * width]
        lines.append(f"  {thesis.summary}")
        lines.append("")
        for condition in thesis.bull_case:
            mark = {"met": "[x]", "unmet": "[ ]", "unknown": "[?]"}[condition.status]
            lines.append(f"  {mark} {condition.claim}")
            lines.append(f"      {condition.current}")
        if thesis.breaks_if:
            lines += ["", "  BREAKS IF:"]
            for item in thesis.breaks_if:
                lines.append(f"    - {item}")
        lines.append("")

    # --- Errors -----------------------------------------------------------
    if report.errors:
        lines += ["-" * width, "  UNAVAILABLE", "-" * width]
        for leg, message in report.errors.items():
            lines.append(f"    {leg}: {message}")
        lines.append("")

    lines += [
        "=" * width,
        "  Research output only: not financial advice, no orders placed.",
        "  nova-research | Nikolay Gelshtein",
        "=" * width,
    ]
    return "\n".join(lines)
