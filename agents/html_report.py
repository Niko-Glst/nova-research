"""HTML rendering of a research report.

Produces a single self-contained file: no external stylesheets, no CDN scripts,
no network calls at view time. Charts are inline SVG generated here rather than
by a charting library, which keeps the output readable offline and avoids
shipping a dependency for what amounts to a handful of bars.

The peer chart is the centrepiece. A table of ratios is hard to scan; the same
numbers as bars against a centre line make "expensive on every multiple, better
on every margin" visible at a glance, which is the actual finding.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from agents.charts import (
    PeerBar,
    equity_curve_chart,
    monte_carlo_histogram,
    peer_metric_chart,
)
from agents.research_report import ResearchReport

# Metric display names, so the page does not show raw snake_case keys.
_METRIC_LABELS = {
    "trailing_pe": "P/E (trailing)",
    "forward_pe": "P/E (forward)",
    "price_to_book": "Price / book",
    "price_to_sales": "Price / sales",
    "peg_ratio": "PEG ratio",
    "profit_margin": "Net margin",
    "operating_margin": "Operating margin",
    "revenue_growth": "Revenue growth",
    "earnings_growth": "Earnings growth",
    "return_on_equity": "Return on equity",
}

# Metrics where being below the peer median is the favourable reading.
_LOWER_IS_BETTER = {
    "trailing_pe", "forward_pe", "price_to_book", "price_to_sales", "peg_ratio",
}

_VERDICT_TONE = {
    "positive": "good",
    "negative": "bad",
    "mixed": "warn",
    "inconclusive": "muted",
}

_SIGNAL_TONE = {
    "bullish": "good", "undervalued": "good",
    "bearish": "bad", "overvalued": "bad",
    "neutral": "warn", "fair": "warn",
    "unknown": "muted",
}


@dataclass(frozen=True)
class _Bar:
    label: str
    ratio: float
    favourable: bool
    own: float
    median: float


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _tone(mapping: dict[str, str], key: str | None) -> str:
    return mapping.get((key or "").lower(), "muted")


def _peer_bars(report: ResearchReport) -> list[_Bar]:
    """Build the peer comparison bars, ordered so valuation metrics lead."""
    fundamental = report.fundamental
    if fundamental is None or fundamental.peer_comparison is None:
        return []

    peers = fundamental.peer_comparison
    bars: list[_Bar] = []

    for metric, ratio in peers.relative.items():
        own = peers.symbol_metrics.get(metric)
        median = peers.peer_medians.get(metric)
        if own is None or median is None:
            continue
        # "Favourable" differs per metric: cheap is good for a multiple, high is
        # good for a margin.
        favourable = ratio < 1.0 if metric in _LOWER_IS_BETTER else ratio > 1.0
        bars.append(
            _Bar(
                label=_METRIC_LABELS.get(metric, metric.replace("_", " ")),
                ratio=ratio,
                favourable=favourable,
                own=own,
                median=median,
            )
        )

    # Valuation multiples first, then quality metrics; alphabetical within each.
    bars.sort(key=lambda b: (b.label not in
                             {_METRIC_LABELS[m] for m in _LOWER_IS_BETTER}, b.label))
    return bars


def _peer_chart_svg(bars: list[_Bar]) -> str:
    """Render peer ratios as a diverging bar chart around a 1.0x centre line."""
    if not bars:
        return ""

    row_height = 34
    label_width = 150
    chart_width = 420
    height = len(bars) * row_height + 46
    centre = label_width + chart_width / 2

    # Clamp the drawn ratio: one 16x outlier would flatten every other bar.
    max_deviation = 2.5
    half = chart_width / 2

    parts = [
        f'<svg viewBox="0 0 {label_width + chart_width + 70} {height}" '
        f'role="img" aria-label="Peer comparison: each metric relative to the peer median" '
        f'class="peer-chart">',
        # Centre line and its label.
        f'<line x1="{centre}" y1="18" x2="{centre}" y2="{height - 22}" '
        f'class="axis"/>',
        f'<text x="{centre}" y="12" class="axis-label" text-anchor="middle">'
        f'peer median (1.0x)</text>',
    ]

    for index, bar in enumerate(bars):
        y = 26 + index * row_height
        # Log-ish compression so both 0.1x and 16x stay on the canvas.
        deviation = max(-max_deviation, min(max_deviation, bar.ratio - 1.0))
        width = abs(deviation) / max_deviation * half
        x = centre if deviation >= 0 else centre - width
        tone = "good" if bar.favourable else "bad"
        clipped = abs(bar.ratio - 1.0) > max_deviation

        parts.append(
            f'<text x="{label_width - 10}" y="{y + 15}" class="bar-label" '
            f'text-anchor="end">{_esc(bar.label)}</text>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 4}" width="{max(width, 1.5):.1f}" height="16" '
            f'class="bar {tone}" rx="2"/>'
        )
        marker = "&#8250;" if clipped else ""
        parts.append(
            f'<text x="{label_width + chart_width + 8}" y="{y + 16}" '
            f'class="bar-value {tone}">{bar.ratio:.2f}x{marker}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _score_gauge_svg(score: float) -> str:
    """A single bar showing where the score sits between -1 and +1."""
    width, height = 320, 44
    centre = width / 2
    clamped = max(-1.0, min(1.0, score))
    bar_width = abs(clamped) * (width / 2 - 10)
    x = centre if clamped >= 0 else centre - bar_width
    tone = "good" if clamped > 0.25 else "bad" if clamped < -0.25 else "warn"

    return (
        f'<svg viewBox="0 0 {width} {height}" class="gauge" role="img" '
        f'aria-label="Score {score:+.2f} on a scale from -1 to +1">'
        f'<line x1="10" y1="{height - 12}" x2="{width - 10}" y2="{height - 12}" class="axis"/>'
        f'<line x1="{centre}" y1="6" x2="{centre}" y2="{height - 8}" class="axis"/>'
        f'<rect x="{x:.1f}" y="10" width="{max(bar_width, 2):.1f}" height="14" '
        f'class="bar {tone}" rx="2"/>'
        f'<text x="10" y="{height - 1}" class="axis-label">-1 bearish</text>'
        f'<text x="{width - 10}" y="{height - 1}" class="axis-label" '
        f'text-anchor="end">+1 bullish</text>'
        f"</svg>"
    )


def _analyst_rows(report: ResearchReport) -> str:
    """One row per analyst: stance, weight, and whether it counted."""
    from agents.research_report import WEIGHTS, _stance_value

    legs = [
        ("Fundamental", report.fundamental.signal if report.fundamental else None, "fundamental"),
        ("Technical", report.technical.signal if report.technical else None, "technical"),
        ("Insider", report.insider.signal if report.insider else None, "insider"),
        ("Sentiment", report.sentiment.signal if report.sentiment else None, "sentiment"),
    ]

    rows = []
    for name, signal, key in legs:
        counted = _stance_value(signal) is not None
        tone = _tone(_SIGNAL_TONE, signal)
        label = (signal or "unavailable").upper()
        note = "counted" if counted else "excluded from score"
        rows.append(
            f"<tr><td>{_esc(name)}</td>"
            f'<td><span class="pill {tone}">{_esc(label)}</span></td>'
            f"<td class=\"num\">{WEIGHTS[key]:.1f}</td>"
            f'<td class="muted-text">{note}</td></tr>'
        )
    return "".join(rows)


def _thesis_section(report: ResearchReport) -> str:
    thesis = report.thesis
    if thesis is None or not thesis.bull_case:
        return ""

    items = []
    for condition in thesis.bull_case:
        tone = {"met": "good", "unmet": "bad", "unknown": "muted"}[condition.status]
        mark = {"met": "&#10003;", "unmet": "&#10007;", "unknown": "?"}[condition.status]
        items.append(
            f'<li class="condition {tone}">'
            f'<span class="mark">{mark}</span>'
            f"<div><strong>{_esc(condition.claim)}</strong>"
            f'<div class="muted-text">{_esc(condition.current)}</div></div></li>'
        )

    breaks = "".join(f"<li>{_esc(item)}</li>" for item in thesis.breaks_if)
    breaks_block = (
        f'<h3>What would break it</h3><ul class="breaks">{breaks}</ul>' if breaks else ""
    )

    return f"""
    <section class="card">
      <h2>Investment thesis</h2>
      <p class="lede">{_esc(thesis.summary)}</p>
      <p class="muted-text">Stance: <strong>{_esc(thesis.stance)}</strong> &mdash;
         {thesis.conditions_met} of {thesis.conditions_total} measurable conditions met.</p>
      <h3>What must be true</h3>
      <ul class="conditions">{"".join(items)}</ul>
      {breaks_block}
    </section>"""


def _swot_section(report: ResearchReport) -> str:
    swot = report.swot
    if swot is None or swot.is_empty:
        return ""

    def quadrant(title: str, items, tone: str) -> str:
        if not items:
            body = '<p class="muted-text">Nothing flagged from the available figures.</p>'
        else:
            body = "".join(
                f"<li><span>{_esc(i.text)}</span>"
                f'<span class="metric-chip">{_esc(i.metric)}: '
                f"<strong>{_esc(i.value)}</strong></span></li>"
                for i in items
            )
            body = f"<ul>{body}</ul>"
        return f'<div class="quadrant {tone}"><h3>{title}</h3>{body}</div>'

    return f"""
    <section class="card">
      <h2>SWOT</h2>
      <p class="muted-text">Derived from the measured figures; each line names the
         number behind it.</p>
      <div class="swot">
        {quadrant("Strengths", swot.strengths, "good")}
        {quadrant("Weaknesses", swot.weaknesses, "bad")}
        {quadrant("Opportunities", swot.opportunities, "good")}
        {quadrant("Threats", swot.threats, "bad")}
      </div>
    </section>"""


def _peer_section(report: ResearchReport) -> str:
    fundamental = report.fundamental
    if fundamental is None or fundamental.peer_comparison is None:
        return ""

    peers = fundamental.peer_comparison
    if not peers.peers_used:
        return (
            f'<section class="card"><h2>Peer comparison</h2>'
            f'<p class="muted-text">{_esc(peers.notes)}</p></section>'
        )

    bars = _peer_bars(report)
    rows = "".join(
        f"<tr><td>{_esc(bar.label)}</td>"
        f'<td class="num">{bar.own:.2f}</td>'
        f'<td class="num">{bar.median:.2f}</td>'
        f'<td class="num {"good" if bar.favourable else "bad"}">{bar.ratio:.2f}x</td></tr>'
        for bar in bars
    )

    scope = "sector" if peers.is_sector_fallback else "niche"
    fallback_note = (
        '<p class="warn-note">No niche-level peer group is declared for this '
        "industry, so this is a broader sector comparison.</p>"
        if peers.is_sector_fallback
        else ""
    )

    return f"""
    <section class="card">
      <h2>Peer comparison &mdash; {_esc(peers.niche)}</h2>
      <p class="muted-text">Compared against {len(peers.peers_used)} {scope} peers:
         {_esc(", ".join(peers.peers_used))}. Bars show each metric relative to the
         peer median; green is the favourable side, which differs per metric
         (cheap is good for a multiple, high is good for a margin).</p>
      {fallback_note}
      {_peer_chart_svg(bars)}
      <table>
        <thead><tr><th>Metric</th><th class="num">This</th>
        <th class="num">Peer median</th><th class="num">Relative</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def _fundamentals_section(report: ResearchReport) -> str:
    fundamental = report.fundamental
    if fundamental is None:
        return ""

    metrics = fundamental.metrics
    display = [
        ("EPS (trailing)", "trailing_eps", "{:.2f}"),
        ("EPS (forward)", "forward_eps", "{:.2f}"),
        ("P/E (trailing)", "trailing_pe", "{:.1f}"),
        ("P/E (forward)", "forward_pe", "{:.1f}"),
        ("Earnings yield", "earnings_yield_pct", "{:.2f}%"),
        ("Net margin", "profit_margin", "{:.1%}"),
        ("Operating margin", "operating_margin", "{:.1%}"),
        ("Revenue growth", "revenue_growth", "{:.1%}"),
        ("Return on equity", "return_on_equity", "{:.1%}"),
        ("Debt / equity", "debt_to_equity", "{:.0f}%"),
        ("FCF yield", "fcf_yield_pct", "{:.2f}%"),
    ]

    cells = []
    for label, key, fmt in display:
        value = metrics.get(key)
        if value is None:
            continue
        cells.append(
            f'<div class="stat"><span class="stat-label">{_esc(label)}</span>'
            f'<span class="stat-value">{fmt.format(value)}</span></div>'
        )

    market_cap = metrics.get("market_cap")
    if market_cap:
        for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if market_cap >= threshold:
                cells.insert(
                    0,
                    f'<div class="stat"><span class="stat-label">Market cap</span>'
                    f'<span class="stat-value">${market_cap / threshold:,.2f}{suffix}'
                    f"</span></div>",
                )
                break

    reasons = "".join(f"<li>{_esc(r)}</li>" for r in fundamental.reasons)
    tone = _tone(_SIGNAL_TONE, fundamental.signal)

    return f"""
    <section class="card">
      <h2>Fundamentals <span class="pill {tone}">{_esc(fundamental.signal.upper())}</span></h2>
      <div class="stats">{"".join(cells)}</div>
      <ul class="reasons">{reasons}</ul>
    </section>"""


def _technical_section(report: ResearchReport) -> str:
    technical = report.technical
    if technical is None:
        return ""

    cells = "".join(
        f'<div class="stat"><span class="stat-label">{_esc(name)}</span>'
        f'<span class="stat-value">{value:,.2f}</span></div>'
        for name, value in technical.indicators.items()
    )
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in technical.reasons)
    tone = _tone(_SIGNAL_TONE, technical.signal)

    return f"""
    <section class="card">
      <h2>Technicals <span class="pill {tone}">{_esc(technical.signal.upper())}</span></h2>
      <div class="stats">{cells}</div>
      <ul class="reasons">{reasons}</ul>
    </section>"""


def _insider_section(report: ResearchReport) -> str:
    insider = report.insider
    if insider is None:
        return ""

    tone = _tone(_SIGNAL_TONE, insider.signal)
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in insider.reasons)

    stats = "".join(
        [
            f'<div class="stat"><span class="stat-label">Open-market buys</span>'
            f'<span class="stat-value">{insider.buy_count}</span></div>',
            f'<div class="stat"><span class="stat-label">Open-market sells</span>'
            f'<span class="stat-value">{insider.sell_count}</span></div>',
            f'<div class="stat"><span class="stat-label">Net flow</span>'
            f'<span class="stat-value">${insider.net_value:,.0f}</span></div>',
            f'<div class="stat"><span class="stat-label">Window</span>'
            f'<span class="stat-value">{insider.lookback_days}d</span></div>',
        ]
    )

    return f"""
    <section class="card">
      <h2>Insider activity <span class="pill {tone}">{_esc(insider.signal.upper())}</span></h2>
      <div class="stats">{stats}</div>
      <ul class="reasons">{reasons}</ul>
    </section>"""


def _sentiment_section(report: ResearchReport) -> str:
    sentiment = report.sentiment
    if sentiment is None:
        return ""

    tone = _tone(_SIGNAL_TONE, sentiment.signal)
    detail = sentiment.news_detail

    stats = [
        f'<div class="stat"><span class="stat-label">News score</span>'
        f'<span class="stat-value">{sentiment.news_score:+.2f}</span></div>',
        f'<div class="stat"><span class="stat-label">Articles</span>'
        f'<span class="stat-value">{sentiment.news_sample}</span></div>',
        f'<div class="stat"><span class="stat-label">Reddit score</span>'
        f'<span class="stat-value">{sentiment.reddit_score:+.2f}</span></div>',
        f'<div class="stat"><span class="stat-label">Reddit posts</span>'
        f'<span class="stat-value">{sentiment.reddit_sample}</span></div>',
    ]

    volume_block = ""
    if detail is not None and detail.volume.status != "unknown":
        volume = detail.volume
        ratio = "no baseline" if volume.ratio == float("inf") else f"{volume.ratio:.2f}x"
        volume_tone = {
            "spike": "warn", "elevated": "warn", "quiet": "muted", "normal": "good",
        }.get(volume.status, "muted")
        volume_block = f"""
        <h3>Coverage volume <span class="pill {volume_tone}">
            {_esc(volume.status.upper())}</span></h3>
        <div class="stats">
          <div class="stat"><span class="stat-label">Last {volume.recent_window_days}d</span>
            <span class="stat-value">{volume.recent_count} ({volume.recent_per_day:.1f}/day)</span></div>
          <div class="stat"><span class="stat-label">Baseline</span>
            <span class="stat-value">{volume.baseline_count} ({volume.baseline_per_day:.1f}/day)</span></div>
          <div class="stat"><span class="stat-label">Ratio</span>
            <span class="stat-value">{ratio}</span></div>
        </div>
        <p class="muted-text">{_esc(volume.notes)}</p>"""

    headlines = ""
    if detail is not None and (detail.top_positive or detail.top_negative):

        def _headline_item(headline, tone: str) -> str:
            meta_parts = []
            if headline.source:
                meta_parts.append(_esc(headline.source))
            if headline.age_days is not None:
                meta_parts.append(
                    "today" if headline.age_days < 1 else f"{headline.age_days:.0f}d ago"
                )
            meta = (
                f'<span class="headline-meta">{" &middot; ".join(meta_parts)}</span>'
                if meta_parts
                else ""
            )
            title = _esc(headline.title)
            # rel="noopener" keeps the opened tab from reaching back into this
            # page; noreferrer avoids leaking the local file path as a referrer.
            body = (
                f'<a href="{_esc(headline.url)}" target="_blank" '
                f'rel="noopener noreferrer">{title}</a>'
                if headline.url
                else title
            )
            return f'<li class="{tone}">{body}{meta}</li>'

        pos = "".join(_headline_item(h, "good") for h in detail.top_positive)
        neg = "".join(_headline_item(h, "bad") for h in detail.top_negative)
        headlines = (
            '<h3>Headlines</h3>'
            f'<ul class="headlines">{pos}{neg}</ul>'
        )

    reasons = "".join(f"<li>{_esc(r)}</li>" for r in sentiment.reasons)

    return f"""
    <section class="card">
      <h2>Sentiment <span class="pill {tone}">{_esc(sentiment.signal.upper())}</span></h2>
      <div class="stats">{"".join(stats)}</div>
      {volume_block}
      {headlines}
      <ul class="reasons">{reasons}</ul>
    </section>"""


def _eps_peer_section(report: ResearchReport) -> str:
    """Per-peer bars for EPS and the headline valuation multiples.

    The relative chart elsewhere shows this company against a median; this shows
    the individual peers, which is what reveals whether that median is a tight
    cluster or an average of extremes.
    """
    fundamental = report.fundamental
    if fundamental is None or fundamental.peer_comparison is None:
        return ""

    peers = fundamental.peer_comparison
    if not peers.peer_values:
        return ""

    charts: list[str] = []
    specs = [
        ("trailing_eps", "Trailing EPS", "{:.2f}"),
        ("forward_pe", "Forward P/E", "{:.1f}"),
        ("profit_margin", "Net margin", "{:.1%}"),
        ("revenue_growth", "Revenue growth", "{:.1%}"),
    ]

    for metric, title, fmt in specs:
        values = peers.peer_values.get(metric)
        own = peers.symbol_metrics.get(metric)
        if own is None:
            own = fundamental.metrics.get(metric)
        if not values or own is None:
            continue

        bars = [PeerBar(label=report.symbol, value=float(own), is_subject=True)]
        bars += [
            PeerBar(label=ticker, value=float(value), is_subject=False)
            for ticker, value in sorted(values.items(), key=lambda kv: -kv[1])
        ]
        charts.append(
            '<div class="chart-block">'
            + peer_metric_chart(bars, title, fmt, peers.peer_medians.get(metric))
            + "</div>"
        )

    if not charts:
        return ""

    return f"""
    <section class="card">
      <h2>Peer detail &mdash; {_esc(peers.niche)}</h2>
      <p class="muted-text">Each peer individually, with {_esc(report.symbol)}
         highlighted. A median hides whether the group is a tight cluster or an
         average of extremes; these bars show which.</p>
      <div class="chart-grid">{"".join(charts)}</div>
    </section>"""


def _simulation_section(report: ResearchReport) -> str:
    """Backtest result, equity curve against buy-and-hold, and the MC histogram."""
    backtest = report.backtest
    if backtest is None:
        return ""

    verdict_tone = "good" if backtest.beats_buy_and_hold else "bad"
    verdict_text = (
        "beats buy &amp; hold" if backtest.beats_buy_and_hold else "trails buy &amp; hold"
    )

    stats = "".join(
        f'<div class="stat"><span class="stat-label">{label}</span>'
        f'<span class="stat-value">{value}</span></div>'
        for label, value in [
            ("Strategy", f"{backtest.total_return_pct:+.1f}%"),
            ("Buy &amp; hold", f"{backtest.buy_and_hold_return_pct:+.1f}%"),
            ("Annualized", f"{backtest.annualized_return_pct:+.1f}%"),
            ("Max drawdown", f"{backtest.max_drawdown_pct:.1f}%"),
            ("Sharpe", f"{backtest.sharpe_ratio:.2f}"),
            ("Trades", f"{backtest.num_trades}"),
            ("Win rate", f"{backtest.win_rate_pct:.0f}%"),
            ("In market", f"{backtest.time_in_market_pct:.0f}%"),
        ]
    )

    equity_block = ""
    if backtest.equity_curve is not None and report.buy_hold_equity is not None:
        equity_block = (
            "<h3>Strategy versus buy &amp; hold</h3>"
            + equity_curve_chart(
                backtest.equity_curve.to_numpy(), report.buy_hold_equity
            )
        )

    mc_block = ""
    simulation = report.monte_carlo
    if simulation is not None and report.simulated_returns is not None:
        percentile_note = ""
        if simulation.observed_percentile is not None:
            where = simulation.observed_percentile
            if where > 90:
                judgement = (
                    "in the top decile of its own resampling, so treat the "
                    "backtest as a favourable draw rather than an expectation"
                )
            elif where < 10:
                judgement = "in the bottom decile: the realised path was a poor draw"
            else:
                judgement = "near the middle: the realised path was typical"
            percentile_note = (
                '<p class="muted-text">The realised backtest sits at percentile '
                f"<strong>{where:.0f}</strong> of this distribution &mdash; {judgement}.</p>"
            )

        mc_stats = "".join(
            f'<div class="stat"><span class="stat-label">{label}</span>'
            f'<span class="stat-value">{value}</span></div>'
            for label, value in [
                ("Median (p50)", f"{simulation.return_percentiles.get('p50', 0):+.1f}%"),
                ("p5", f"{simulation.return_percentiles.get('p5', 0):+.1f}%"),
                ("p95", f"{simulation.return_percentiles.get('p95', 0):+.1f}%"),
                ("P(loss)", f"{simulation.probability_of_loss:.0%}"),
                ("P(DD &gt; 20%)", f"{simulation.probability_of_drawdown_20pct:.0%}"),
                ("CVaR 5%", f"{simulation.conditional_var_5pct:.1f}%"),
            ]
        )

        histogram = monte_carlo_histogram(
            report.simulated_returns,
            observed=simulation.observed_return_pct,
            percentiles=simulation.return_percentiles,
        )

        mc_block = f"""
        <h3>Monte Carlo &mdash; {simulation.num_simulations:,} resampled paths</h3>
        <p class="muted-text">The backtest gives one path: the one that happened.
           Resampling its returns shows the distribution it could plausibly have
           produced over a {simulation.horizon_days}-day horizon.</p>
        <div class="stats">{mc_stats}</div>
        {histogram}
        {percentile_note}"""

    return f"""
    <section class="card">
      <h2>Backtest <span class="pill {verdict_tone}">{verdict_text}</span></h2>
      <p class="muted-text">50/200-day crossover, 10bps per side, signals shifted
         one bar so they are traded after they are observed. Simulation only.</p>
      <div class="stats">{stats}</div>
      {equity_block}
      {mc_block}
    </section>"""


_STYLE = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #1a1d21; --muted: #6b7280;
  --line: #e3e6ea; --good: #1a7f4b; --good-bg: #e6f4ec;
  --bad: #b3261e; --bad-bg: #fbeae9; --warn: #8a6100; --warn-bg: #fdf3e0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 900px; margin: 0 auto; padding: 28px 20px 60px; }
header { margin-bottom: 22px; }
h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -0.5px; }
h2 { font-size: 17px; margin: 0 0 14px; display: flex; align-items: center; gap: 10px; }
h3 { font-size: 14px; margin: 20px 0 8px; text-transform: uppercase;
  letter-spacing: 0.6px; color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 16px; }
.verdict-head { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.verdict { font-size: 26px; font-weight: 700; letter-spacing: -0.3px; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 99px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.pill.good, .num.good, li.good, .bar-value.good { color: var(--good); }
.pill.good { background: var(--good-bg); }
.pill.bad, .num.bad, li.bad, .bar-value.bad { color: var(--bad); }
.pill.bad { background: var(--bad-bg); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }
.pill.muted { background: #eceef1; color: var(--muted); }
.lede { font-size: 16px; margin: 0 0 10px; }
.muted-text { color: var(--muted); font-size: 13px; }
.warn-note { background: var(--warn-bg); color: var(--warn); padding: 8px 12px;
  border-radius: 6px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; }
.stat { background: #fafbfc; border: 1px solid var(--line); border-radius: 7px;
  padding: 8px 12px; min-width: 118px; flex: 1 1 118px; }
.stat-label { display: block; font-size: 11px; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.4px; }
.stat-value { display: block; font-size: 17px; font-weight: 600;
  font-variant-numeric: tabular-nums; margin-top: 2px; }
.reasons, .breaks, .headlines { margin: 14px 0 0; padding-left: 18px; font-size: 14px; }
.reasons li, .breaks li, .headlines li { margin-bottom: 5px; }
.conditions { list-style: none; margin: 0; padding: 0; }
.condition { display: flex; gap: 11px; padding: 10px 0;
  border-bottom: 1px solid var(--line); }
.condition:last-child { border-bottom: none; }
.condition .mark { font-weight: 700; font-size: 15px; flex-shrink: 0; width: 16px; }
.condition.good .mark { color: var(--good); }
.condition.bad .mark { color: var(--bad); }
.condition.muted .mark { color: var(--muted); }
.swot { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.quadrant { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.quadrant.good { background: var(--good-bg); border-color: #cfe6da; }
.quadrant.bad { background: var(--bad-bg); border-color: #f0d4d2; }
.quadrant h3 { margin-top: 0; color: inherit; }
.quadrant ul { margin: 0; padding-left: 16px; font-size: 13px; }
.quadrant li { margin-bottom: 7px; }
.metric-chip { display: block; font-size: 11px; color: var(--muted); }
.peer-chart, .gauge { width: 100%; height: auto; max-width: 640px; margin: 8px 0 4px; }
.axis { stroke: #c6cad0; stroke-width: 1; }
.axis-label { font-size: 10px; fill: var(--muted); }
.bar-label { font-size: 12px; fill: var(--ink); }
.bar-value { font-size: 11px; font-variant-numeric: tabular-nums; }
.bar.good { fill: var(--good); }
.bar.bad { fill: var(--bad); }
.bar.warn { fill: #d09000; }
footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 26px; }
.brand { font-size: 10px; font-weight: 700; letter-spacing: 2.2px;
  color: var(--muted); margin-bottom: 6px; }
.brand-footer { margin-top: 10px; font-size: 11px; font-weight: 600;
  letter-spacing: 1.6px; color: var(--muted); opacity: 0.85; }
.headlines a { color: inherit; text-decoration: none;
  border-bottom: 1px solid currentColor; }
.headlines a:hover { opacity: 0.75; }
.headline-meta { display: block; font-size: 11px; color: var(--muted);
  letter-spacing: 0.2px; margin-top: 1px; }
@media (max-width: 620px) { .swot { grid-template-columns: 1fr; } }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.chart-block { min-width: 0; }
.chart-title { font-size: 12px; font-weight: 600; fill: var(--ink); }
.median-line { stroke: #9aa0a6; stroke-width: 1; stroke-dasharray: 3 3; }
.bar.subject { fill: #1f4fd8; }
.bar.pos { fill: #7c9cc4; }
.bar.neg { fill: #d09a96; }
.bar-label.subject { font-weight: 700; fill: #1f4fd8; }
.mc-chart, .equity-chart { width: 100%; height: auto; margin: 10px 0; }
.hist.pos { fill: var(--good); opacity: 0.72; }
.hist.neg { fill: var(--bad); opacity: 0.72; }
.marker { stroke-width: 1.5; }
.marker.zero { stroke: #6b7280; stroke-dasharray: 3 3; }
.marker.p5 { stroke: var(--bad); }
.marker.observed { stroke: #1f4fd8; }
.marker-label { font-size: 10px; font-weight: 600; }
.marker-label.zero { fill: #6b7280; }
.marker-label.p5 { fill: var(--bad); }
.marker-label.observed { fill: #1f4fd8; }
.curve { fill: none; stroke-width: 1.8; }
.curve.strategy { stroke: #1f4fd8; }
.curve.buyhold { stroke: #9aa0a6; stroke-dasharray: 4 3; }
.curve-label { font-size: 11px; font-weight: 600; }
.curve-label.strategy { fill: #1f4fd8; }
.curve-label.buyhold { fill: #6b7280; }
@media (max-width: 620px) { .chart-grid { grid-template-columns: 1fr; } }
"""


def render_report(report: ResearchReport, generated_at: str = "") -> str:
    """Render one research report as a standalone HTML document."""
    tone = _tone(_VERDICT_TONE, report.verdict)
    stamp = f"<p class=\"muted-text\">Generated {_esc(generated_at)}</p>" if generated_at else ""

    errors_block = ""
    if report.errors:
        rows = "".join(
            f"<li><strong>{_esc(k)}</strong>: {_esc(v)}</li>"
            for k, v in report.errors.items()
        )
        errors_block = (
            f'<section class="card"><h2>Unavailable data</h2>'
            f'<ul class="reasons">{rows}</ul></section>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(report.symbol)} research report &middot; Nikolay Gelshtein</title>
<style>{_STYLE}</style></head>
<body><div class="wrap">
<header>
  <div class="brand">NIKOLAY GELSHTEIN</div>
  <h1>{_esc(report.symbol)}</h1>
  {stamp}
</header>

<section class="card">
  <div class="verdict-head">
    <span class="verdict">{_esc(report.verdict.upper())}</span>
    <span class="pill {tone}">score {report.score:+.2f}</span>
    <span class="pill muted">confidence {report.confidence:.0%}</span>
  </div>
  {_score_gauge_svg(report.score)}
  <table>
    <thead><tr><th>Analyst</th><th>Stance</th><th class="num">Weight</th><th></th></tr></thead>
    <tbody>{_analyst_rows(report)}</tbody>
  </table>
</section>

{_thesis_section(report)}
{_swot_section(report)}
{_peer_section(report)}
{_eps_peer_section(report)}
{_simulation_section(report)}
{_fundamentals_section(report)}
{_technical_section(report)}
{_insider_section(report)}
{_sentiment_section(report)}
{errors_block}

<footer>
  Research output only &mdash; not financial advice, and no orders are placed.<br>
  Figures are point-in-time reads from public data sources and may be stale or wrong.
  <div class="brand-footer">nova-research &middot; Nikolay Gelshtein</div>
</footer>
</div></body></html>"""
