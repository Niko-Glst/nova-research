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


def _nice_ceiling(value: float) -> float:
    """Round a magnitude up to a readable axis bound (1, 2, 2.5 or 5 x 10^n)."""
    if value <= 0 or not math.isfinite(value):
        return 1.0
    exponent = math.floor(math.log10(value))
    base = value / (10**exponent)
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if base <= candidate:
            return candidate * (10**exponent)
    return 10.0 * (10**exponent)


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
