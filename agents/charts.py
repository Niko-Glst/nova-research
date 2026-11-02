"""Inline-SVG charts for the HTML report.

Written by hand rather than with a plotting library for the same reason the rest
of the page is: the output has to be a single self-contained file that opens
offline, and matplotlib would mean either a PNG (unreadable at other zoom
levels, no text selection) or a heavy dependency that already proved brittle in
this project.

Each function returns an `<svg>` element as a string. They share the CSS classes
defined in html_report._STYLE, so colours and fonts stay consistent with the
page and follow the light/dark treatment there.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import numpy as np


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------
# EPS / peer bars
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerBar:
    """One company's value for a metric, in a peer comparison."""

    label: str
    value: float
    is_subject: bool  # True for the analyzed company, drawn differently


def peer_metric_chart(
    bars: list[PeerBar],
    title: str,
    value_format: str = "{:.2f}",
    median: float | None = None,
) -> str:
    """A horizontal bar per company, with the analyzed one highlighted.

    Handles negative values (a loss-making peer has negative EPS) by placing the
    zero line where it belongs rather than at the left edge.
    """
    if not bars:
        return ""

    row_height = 30
    label_width = 78
    chart_width = 360
    value_width = 74
    height = len(bars) * row_height + 44
    width = label_width + chart_width + value_width

    values = [b.value for b in bars]
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    span = high - low or 1.0

    def x_for(value: float) -> float:
        return label_width + (value - low) / span * chart_width

    zero_x = x_for(0.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="peer-chart" role="img" '
        f'aria-label="{_esc(title)} compared across peers">',
        f'<text x="{label_width}" y="12" class="chart-title">{_esc(title)}</text>',
        f'<line x1="{zero_x:.1f}" y1="20" x2="{zero_x:.1f}" y2="{height - 20}" class="axis"/>',
    ]

    if median is not None and low <= median <= high:
        median_x = x_for(median)
        parts.append(
            f'<line x1="{median_x:.1f}" y1="20" x2="{median_x:.1f}" '
            f'y2="{height - 20}" class="median-line"/>'
        )
        parts.append(
            f'<text x="{median_x:.1f}" y="{height - 8}" class="axis-label" '
            f'text-anchor="middle">peer median</text>'
        )

    for index, bar in enumerate(bars):
        y = 24 + index * row_height
        bar_x = zero_x if bar.value >= 0 else x_for(bar.value)
        bar_width = abs(x_for(bar.value) - zero_x)
        tone = "subject" if bar.is_subject else ("pos" if bar.value >= 0 else "neg")

        parts.append(
            f'<text x="{label_width - 8}" y="{y + 15}" class="bar-label'
            f'{" subject" if bar.is_subject else ""}" text-anchor="end">'
            f"{_esc(bar.label)}</text>"
        )
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{y + 4}" width="{max(bar_width, 1.5):.1f}" '
            f'height="15" class="bar {tone}" rx="2"/>'
        )
        parts.append(
            f'<text x="{label_width + chart_width + 6}" y="{y + 16}" '
            f'class="bar-value">{_esc(value_format.format(bar.value))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Monte Carlo distribution
# --------------------------------------------------------------------------


def monte_carlo_histogram(
    terminal_returns: np.ndarray,
    observed: float | None = None,
    percentiles: dict[str, float] | None = None,
    bins: int = 44,
) -> str:
    """Histogram of simulated terminal returns, with markers.

    The vertical lines are what make this readable: zero (break-even), the 5th
    percentile (the loss you should be prepared for), and where the realised
    backtest actually landed.
    """
    values = np.asarray(terminal_returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return ""

    # Clip the extreme tails so a handful of outliers don't flatten the body.
    low, high = np.percentile(values, [0.5, 99.5])
    if high <= low:
        low, high = values.min(), values.max() + 1e-9

    counts, edges = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    peak = counts.max() or 1

    width, height = 640, 230
    left, right, top, bottom = 46, 14, 18, 34
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_for(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="mc-chart" role="img" '
        f'aria-label="Distribution of {values.size} simulated terminal returns">',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" class="axis"/>',
    ]

    bar_width = plot_width / bins
    for index, count in enumerate(counts):
        if count == 0:
            continue
        bar_height = count / peak * plot_height
        x = left + index * bar_width
        y = top + plot_height - bar_height
        # Colour by sign: losses in red, gains in green.
        centre = (edges[index] + edges[index + 1]) / 2
        tone = "neg" if centre < 0 else "pos"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 0.6, 0.6):.1f}" '
            f'height="{bar_height:.1f}" class="hist {tone}"/>'
        )

    def marker(value: float, label: str, css: str) -> None:
        if not (low <= value <= high):
            return
        x = x_for(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{top + plot_height}" class="marker {css}"/>'
        )
        anchor = "start" if x < left + plot_width * 0.75 else "end"
        offset = 4 if anchor == "start" else -4
        parts.append(
            f'<text x="{x + offset:.1f}" y="{top + 11}" class="marker-label {css}" '
            f'text-anchor="{anchor}">{_esc(label)}</text>'
        )

    marker(0.0, "break-even", "zero")
    if percentiles and "p5" in percentiles:
        marker(percentiles["p5"], f"p5 {percentiles['p5']:.0f}%", "p5")
    if observed is not None:
        marker(observed, f"observed {observed:.1f}%", "observed")

    # X axis labels at both ends and the middle.
    for value in (low, (low + high) / 2, high):
        x = x_for(value)
        anchor = "start" if value == low else ("end" if value == high else "middle")
        parts.append(
            f'<text x="{x:.1f}" y="{height - 14}" class="axis-label" '
            f'text-anchor="{anchor}">{value:.0f}%</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 2}" class="axis-label" '
        f'text-anchor="middle">terminal return over the horizon</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Equity curve vs buy-and-hold
# --------------------------------------------------------------------------


def equity_curve_chart(
    strategy_equity: np.ndarray,
    buy_hold_equity: np.ndarray,
    labels: tuple[str, str] = ("Strategy", "Buy & hold"),
) -> str:
    """Two normalized equity curves on one axis.

    Both start at 1.0, so the vertical distance between them at any point is the
    cumulative difference in outcome -- which is the comparison that decides
    whether a strategy was worth running at all.
    """
    strategy = np.asarray(strategy_equity, dtype=float)
    buy_hold = np.asarray(buy_hold_equity, dtype=float)
    if strategy.size < 2 or buy_hold.size < 2:
        return ""

    n = min(strategy.size, buy_hold.size)
    strategy, buy_hold = strategy[:n], buy_hold[:n]

    width, height = 640, 250
    left, right, top, bottom = 48, 90, 16, 30
    plot_width = width - left - right
    plot_height = height - top - bottom

    low = float(min(strategy.min(), buy_hold.min(), 1.0))
    high = float(max(strategy.max(), buy_hold.max(), 1.0))
    pad = (high - low) * 0.06 or 0.1
    low, high = low - pad, high + pad

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + index / (n - 1) * plot_width
        y = top + plot_height - (value - low) / (high - low) * plot_height
        return x, y

    def path_for(series: np.ndarray) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(point(i, v) for i, v in enumerate(series))
        )

    _, baseline_y = point(0, 1.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="equity-chart" role="img" '
        f'aria-label="Strategy equity curve against buy and hold">',
        f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{left + plot_width}" '
        f'y2="{baseline_y:.1f}" class="axis"/>',
        f'<path d="{path_for(buy_hold)}" class="curve buyhold"/>',
        f'<path d="{path_for(strategy)}" class="curve strategy"/>',
    ]

    # End-of-series labels, nudged apart when the curves finish close together.
    end_strategy = point(n - 1, float(strategy[-1]))
    end_buy_hold = point(n - 1, float(buy_hold[-1]))
    if abs(end_strategy[1] - end_buy_hold[1]) < 13:
        if end_strategy[1] <= end_buy_hold[1]:
            end_strategy = (end_strategy[0], end_strategy[1] - 7)
            end_buy_hold = (end_buy_hold[0], end_buy_hold[1] + 7)
        else:
            end_strategy = (end_strategy[0], end_strategy[1] + 7)
            end_buy_hold = (end_buy_hold[0], end_buy_hold[1] - 7)

    parts.append(
        f'<text x="{left + plot_width + 6}" y="{end_strategy[1]:.1f}" '
        f'class="curve-label strategy">{_esc(labels[0])} '
        f"{(strategy[-1] - 1) * 100:+.0f}%</text>"
    )
    parts.append(
        f'<text x="{left + plot_width + 6}" y="{end_buy_hold[1]:.1f}" '
        f'class="curve-label buyhold">{_esc(labels[1])} '
        f"{(buy_hold[-1] - 1) * 100:+.0f}%</text>"
    )

    for value in (low + pad, 1.0, high - pad):
        _, y = point(0, value)
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" class="axis-label" '
            f'text-anchor="end">{value:.2f}x</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Outcome probabilities
# --------------------------------------------------------------------------


def probability_ladder_chart(
    thresholds: list[float],
    by_regime: dict[str, list[float]],
    title: str,
    positive_is_good: bool = True,
) -> str:
    """Grouped horizontal bars: one threshold per row, one bar per regime.

    Bars are drawn on a fixed 0-100% scale so rows are comparable across the
    two ladders. Values are printed inside the bar when it is wide enough to
    hold the text and outside otherwise, which is what keeps labels off each
    other at small probabilities.
    """
    if not thresholds or not by_regime:
        return ""

    regimes = list(by_regime)
    bar_height = 13
    bar_gap = 2
    group_gap = 12
    group_height = len(regimes) * (bar_height + bar_gap) + group_gap

    label_width = 58
    chart_width = 300
    right_gutter = 44
    header = 34
    footer = 22

    height = len(thresholds) * group_height + header + footer
    width = label_width + chart_width + right_gutter

    tones = {"calm": "calm", "volatile": "volatile", "all": "blend"}

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="ladder-chart" role="img" '
        f'aria-label="{_esc(title)} by volatility regime">',
        f'<text x="0" y="11" class="chart-title">{_esc(title)}</text>',
    ]

    # Legend, right-aligned on the title row.
    legend_x = width
    for regime in reversed(regimes):
        legend_x -= 58
        parts.append(
            f'<rect x="{legend_x}" y="3" width="8" height="8" '
            f'class="bar {tones.get(regime, "blend")}" rx="1"/>'
        )
        parts.append(
            f'<text x="{legend_x + 11}" y="11" class="axis-label">{_esc(regime)}</text>'
        )

    # Gridlines at 0, 25, 50, 75, 100%.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = label_width + fraction * chart_width
        parts.append(
            f'<line x1="{x:.1f}" y1="{header - 8}" x2="{x:.1f}" '
            f'y2="{height - footer + 2}" class="gridline"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - footer + 15}" class="axis-label" '
            f'text-anchor="middle">{fraction * 100:.0f}%</text>'
        )

    for row, threshold in enumerate(thresholds):
        group_y = header + row * group_height
        sign = "+" if positive_is_good and threshold >= 0 else "\u2212"
        label = f"{sign}{abs(threshold):.0f}%"
        centre = group_y + (len(regimes) * (bar_height + bar_gap)) / 2

        parts.append(
            f'<text x="{label_width - 10}" y="{centre + 3:.1f}" '
            f'class="row-label" text-anchor="end">{label}</text>'
        )

        for index, regime in enumerate(regimes):
            probability = by_regime[regime][row]
            bar_y = group_y + index * (bar_height + bar_gap)
            bar_width = probability * chart_width
            tone = tones.get(regime, "blend")

            parts.append(
                f'<rect x="{label_width}" y="{bar_y}" '
                f'width="{max(bar_width, 0.8):.1f}" height="{bar_height}" '
                f'class="bar {tone}" rx="1"/>'
            )
            # Inside the bar when there is room; outside when there is not.
            if bar_width >= 34:
                parts.append(
                    f'<text x="{label_width + bar_width - 5:.1f}" '
                    f'y="{bar_y + bar_height - 3}" class="bar-inline" '
                    f'text-anchor="end">{probability:.0%}</text>'
                )
            else:
                parts.append(
                    f'<text x="{label_width + bar_width + 5:.1f}" '
                    f'y="{bar_y + bar_height - 3}" class="bar-value">'
                    f'{probability:.0%}</text>'
                )

    parts.append("</svg>")
    return "".join(parts)


def outcome_fan_chart(
    percentiles_by_regime: dict[str, dict[int, float]],
    horizon_days: int,
) -> str:
    """Percentile ranges as horizontal bands, one row per regime.

    Each band spans p5 to p95, with p25-p75 shaded darker and the median
    marked. Endpoint values sit on the axis row beneath all bands rather than
    under each one, which is what previously caused rows to collide.
    """
    if not percentiles_by_regime:
        return ""

    regimes = list(percentiles_by_regime)
    band_height = 20
    row_height = 30
    label_width = 62
    chart_width = 430
    header = 26
    axis_height = 46

    height = len(regimes) * row_height + header + axis_height
    width = label_width + chart_width + 30

    every = [v for p in percentiles_by_regime.values() for v in p.values()]
    low, high = min(every), max(every)
    low = min(low, 0.0)
    pad = ((high - low) or 1.0) * 0.08
    low, high = low - pad, high + pad
    span = high - low

    def x_for(value: float) -> float:
        return label_width + (value - low) / span * chart_width

    zero_x = x_for(0.0)
    tones = {"calm": "calm", "volatile": "volatile", "all": "blend"}

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="fan-chart" role="img" '
        f'aria-label="Return percentile ranges over {horizon_days} days by regime">',
    ]

    # Zero reference line spanning the plot area.
    plot_bottom = header + len(regimes) * row_height
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{header - 8}" x2="{zero_x:.1f}" '
        f'y2="{plot_bottom + 4}" class="zero-line"/>'
    )
    parts.append(
        f'<text x="{zero_x:.1f}" y="{header - 12}" class="axis-label" '
        f'text-anchor="middle">0%</text>'
    )

    for index, regime in enumerate(regimes):
        p = percentiles_by_regime[regime]
        y = header + index * row_height
        tone = tones.get(regime, "blend")

        outer_x, outer_w = x_for(p[5]), x_for(p[95]) - x_for(p[5])
        inner_x, inner_w = x_for(p[25]), x_for(p[75]) - x_for(p[25])
        median_x = x_for(p[50])

        parts.append(
            f'<text x="{label_width - 10}" y="{y + band_height - 6}" '
            f'class="row-label" text-anchor="end">{_esc(regime)}</text>'
        )
        parts.append(
            f'<rect x="{outer_x:.1f}" y="{y}" width="{max(outer_w, 1):.1f}" '
            f'height="{band_height}" class="band outer {tone}" rx="2"/>'
        )
        parts.append(
            f'<rect x="{inner_x:.1f}" y="{y}" width="{max(inner_w, 1):.1f}" '
            f'height="{band_height}" class="band inner {tone}" rx="2"/>'
        )
        parts.append(
            f'<line x1="{median_x:.1f}" y1="{y - 2}" x2="{median_x:.1f}" '
            f'y2="{y + band_height + 2}" class="median-mark"/>'
        )
        # Median value to the right of the band, clear of every other label.
        parts.append(
            f'<text x="{label_width + chart_width + 6}" '
            f'y="{y + band_height - 6}" class="bar-value">{p[50]:+.0f}%</text>'
        )

    # One axis for all bands: ticks at the extremes and the midpoint.
    axis_y = plot_bottom + 10
    parts.append(
        f'<line x1="{label_width}" y1="{axis_y}" '
        f'x2="{label_width + chart_width}" y2="{axis_y}" class="axis"/>'
    )
    for value in (low + pad, (low + high) / 2, high - pad):
        x = x_for(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" '
            f'y2="{axis_y + 4}" class="axis"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{axis_y + 15}" class="axis-label" '
            f'text-anchor="middle">{value:+.0f}%</text>'
        )
    parts.append(
        f'<text x="{label_width + chart_width / 2}" y="{axis_y + 27}" '
        f'class="axis-label" text-anchor="middle">'
        f'{horizon_days}-day return. Band: p5 to p95. Shaded: p25 to p75. '
        f'Line: median.</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Market breadth: RSP/SPY with zoomable views
# --------------------------------------------------------------------------


def _ratio_view_svg(window, ma, trend, low: float, high: float, zone: float,
                    label: str, turn_ma: int) -> str:
    """One view of the ratio: zone bands, the ratio, its turn average and trend line."""
    import pandas as pd  # local: the rest of this module works on numpy arrays

    width, height = 640, 250
    left, right, top, bottom = 52, 104, 14, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [window.to_numpy(dtype=float)]
    for extra in (ma, trend):
        if len(extra):
            values.append(extra.to_numpy(dtype=float))
    data = np.concatenate(values)
    data = data[np.isfinite(data)]
    lo, hi = float(data.min()), float(data.max())
    if label == "1Y":  # the full range and its zones belong in the yearly view
        lo, hi = min(lo, low), max(hi, high)
    pad = (hi - lo) * 0.06 or 0.001
    lo, hi = lo - pad, hi + pad
    dates = list(window.index)
    n = len(dates)

    def x(i: float) -> float:
        return left + (i / max(n - 1, 1)) * plot_w

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    position = {d: i for i, d in enumerate(dates)}

    def path(series) -> str:
        pts = [(position[d], v) for d, v in series.items() if d in position and np.isfinite(v)]
        return " ".join(f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}"
                        for k, (i, v) in enumerate(pts))

    clip = f"bz-clip-{label}"
    band = (high - low) * zone
    parts = [f'<svg viewBox="0 0 {width} {height}" class="bz-chart" role="img" '
             f'aria-label="Equal-weight over cap-weight S&amp;P 500, last {_esc(label)}">',
             f'<defs><clipPath id="{clip}"><rect x="{left}" y="{top}" width="{plot_w}" '
             f'height="{plot_h}"/></clipPath></defs><g clip-path="url(#{clip})">']
    for z_lo, z_hi, name in ((low, low + band, "bottom zone"), (high - band, high, "top zone")):
        y1, y2 = y(z_hi), y(z_lo)
        parts.append(f'<rect x="{left}" y="{y1:.1f}" width="{plot_w}" height="{max(y2 - y1, 0):.1f}" '
                     f'class="bz-zone"><title>{name} of the 1-year range: {z_lo:.4f} to '
                     f'{z_hi:.4f}</title></rect>')
    for level in (low, high):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(level):.1f}" '
                     f'y2="{y(level):.1f}" class="bz-range"/>')
    parts.append("</g>")
    for k in range(4):
        v = lo + pad + k * (hi - lo - 2 * pad) / 3
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                     f'class="bz-grid"/><text x="{left - 6}" y="{y(v) + 3.5:.1f}" '
                     f'class="axis-label" text-anchor="end">{v:.3f}</text>')
    step = {"1Y": 21, "3M": 10, "1M": 5}.get(label, max(n // 6, 1))
    for i in range(0, n, step):
        fmt = "%b" if label == "1Y" else "%d %b"
        parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" class="axis-label" '
                     f'text-anchor="middle">{pd.Timestamp(dates[i]).strftime(fmt)}</text>')
    if len(ma):
        parts.append(f'<path d="{path(ma)}" class="bz-ma"/>')
    if len(trend):
        parts.append(f'<path d="{path(trend)}" class="bz-trend"/>')
    parts.append(f'<path d="{path(window)}" class="bz-ratio"/>')
    last = float(window.iloc[-1])
    parts.append(f'<circle cx="{x(n - 1):.1f}" cy="{y(last):.1f}" r="4" class="bz-dot"/>')
    parts.append(f'<text x="{x(n - 1) + 8:.1f}" y="{y(last) + 4:.1f}" class="curve-label bz-end">'
                 f'RSP/SPY {last:.4f}</text>')
    slot = plot_w / max(n, 1)
    for i, d in enumerate(dates):
        m = ma.get(d, np.nan) if len(ma) else np.nan
        tip = f"{pd.Timestamp(d).date()}: RSP/SPY {float(window.iloc[i]):.4f}"
        if np.isfinite(m):
            tip += f", {turn_ma}-day {m:.4f}"
        parts.append(f'<rect x="{x(i) - slot / 2:.1f}" y="{top}" width="{slot:.2f}" '
                     f'height="{plot_h}" class="bz-hover"><title>{_esc(tip)}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def breadth_ratio_chart(history, low: float, high: float, zone: float, turn_ma: int,
                        trend, views: dict[str, int], trend_label: str = "Trend line") -> str:
    """RSP/SPY over the last year, with 3-month and 1-month zoom buttons (no script).

    `history` is the ratio (date index) with at least `turn_ma` extra days for the
    moving average; `trend` the fitted trend line (date index).
    """
    import pandas as pd

    if history is None or len(history) < 2:
        return ""
    ma = history.rolling(turn_ma).mean()
    trend = pd.Series(trend) if trend is not None else pd.Series(dtype=float)
    labels = list(views)
    default = labels[0]
    radios, panels, css = [], [], []
    for label in labels:
        days = views[label]
        window = history.iloc[-days:]
        svg = _ratio_view_svg(window, ma.loc[window.index].dropna(),
                              trend[trend.index.isin(window.index)], low, high, zone, label,
                              turn_ma)
        ident = f"bz-{label}"
        radios.append(f'<input type="radio" name="bz-view" id="{ident}" class="bz-radio"'
                      f'{" checked" if label == default else ""}>'
                      f'<label for="{ident}" class="bz-tab">{_esc(label)}</label>')
        panels.append(f'<div class="bz-view bz-v-{label}">{svg}</div>')
        css.append(f"#{ident}:checked ~ .bz-views .bz-v-{label} {{ display: block; }}")
    legend = ('<div class="bz-legend"><span><i class="bz-key bz-key-ratio"></i>RSP/SPY</span>'
              f'<span><i class="bz-key bz-key-ma"></i>{turn_ma}-day average</span>'
              f'<span><i class="bz-key bz-key-trend"></i>{_esc(trend_label)}</span>'
              f'<span><i class="bz-key bz-key-zone"></i>Top / bottom {zone:.0%} of the 1-year '
              'range</span></div>')
    return (f'<style>{" ".join(css)}</style><div class="bz">{"".join(radios)}{legend}'
            f'<div class="bz-views">{"".join(panels)}</div></div>')


# --------------------------------------------------------------------------
# Market move against the implied move (VIX/16)
# --------------------------------------------------------------------------


def implied_move_chart(recent, noise_max: float, notable_max: float,
                       partial_last: bool = False) -> str:
    """One bar per session (its move) against that session's +-1x and +-2x implied bands.

    `recent` has a date index and the columns move, implied, vol_prev, multiple, tier.
    """
    import pandas as pd

    if recent is None or len(recent) == 0:
        return ""
    width, height = 640, 230
    left, right, top, bottom = 52, 14, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    moves = recent["move"].to_numpy(dtype=float) * 100.0
    implied = recent["implied"].to_numpy(dtype=float) * 100.0
    reach = float(np.nanmax(np.concatenate([np.abs(moves), implied * notable_max])))
    bound = max(math.ceil(reach * 1.08 * 2) / 2, 0.5)  # half-percent steps
    n = len(recent)
    slot = plot_w / n
    bar_w = max(slot * 0.56, 2.0)

    def y(v: float) -> float:
        return top + (bound - v) / (2 * bound) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="im-chart" role="img" '
             f'aria-label="S&amp;P 500 daily moves against the move VIX implied, last {n} '
             f'sessions">']
    for k in (-bound, -bound / 2, 0.0, bound / 2, bound):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(k):.1f}" y2="{y(k):.1f}" '
                     f'class="{"zero-line" if k == 0 else "gridline"}"/>'
                     f'<text x="{left - 6}" y="{y(k) + 3.5:.1f}" class="axis-label" '
                     f'text-anchor="end">{k:+.1f}%</text>')
    for i, (day, row) in enumerate(recent.iterrows()):
        x0 = left + i * slot
        band = float(row["implied"]) * 100.0
        for mult, cls in ((notable_max, "im-band2"), (noise_max, "im-band1")):
            parts.append(f'<rect x="{x0 + 1:.1f}" y="{y(band * mult):.1f}" '
                         f'width="{slot - 2:.1f}" height="{y(-band * mult) - y(band * mult):.1f}" '
                         f'class="{cls}"/>')
        move = float(row["move"]) * 100.0
        tier = str(row["tier"])
        cls = f"im-bar {tier}" + (" down" if move < 0 else " up")
        last = i == n - 1
        if last and partial_last:
            cls += " partial"
        y_top, y_bot = (y(move), y(0.0)) if move >= 0 else (y(0.0), y(move))
        parts.append(f'<rect x="{x0 + (slot - bar_w) / 2:.1f}" y="{y_top:.1f}" '
                     f'width="{bar_w:.1f}" height="{max(y_bot - y_top, 1.0):.1f}" class="{cls}"/>')
        if i % 5 == 0 or last:
            parts.append(f'<text x="{x0 + slot / 2:.1f}" y="{height - 8}" class="axis-label" '
                         f'text-anchor="middle">{pd.Timestamp(day).strftime("%d %b")}</text>')
        tip = (f"{pd.Timestamp(day).date()}{' (so far)' if last and partial_last else ''}: "
               f"S&P {move:+.2f}%, implied ±{band:.2f}% (VIX {float(row['vol_prev']):.2f}), "
               f"{abs(float(row['multiple'])):.2f}x, {tier}")
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{slot:.2f}" height="{plot_h}" '
                     f'class="bz-hover"><title>{_esc(tip)}</title></rect>')
    parts.append("</svg>")
    legend = (
        '<div class="bz-legend">'
        f'<span><i class="bz-key im-key-band1"></i>within {noise_max:g}x implied (noise)</span>'
        f'<span><i class="bz-key im-key-band2"></i>{noise_max:g}x to {notable_max:g}x</span>'
        '<span><i class="bz-key im-key-notable"></i>notable move</span>'
        f'<span><i class="bz-key im-key-beyond"></i>drop beyond {notable_max:g}x</span>'
        '</div>')
    return legend + "".join(parts)
