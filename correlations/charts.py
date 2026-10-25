"""Inline-SVG charts for the correlation report.

Same approach as agents/charts.py: hand-written SVG so the report is one
self-contained file that opens offline, with no plotting dependency. Colours
are CSS classes resolved by custom properties (see CHART_CSS), so light and
dark mode each get their own validated steps rather than an automatic flip.

Colour jobs, following the project's data-visualisation method:

- **Correlation (heatmaps):** diverging. Blue for negative, red for positive,
  a neutral grey at zero, binned in steps of 0.1 so a cell's colour maps to a
  readable range. Positive is red because co-movement is the risk being
  measured.
- **Key pairs (lines, scatter):** categorical slots 1-3 (blue, orange, aqua),
  validated as an all-pairs set in both modes. Aqua is under 3:1 contrast on
  the light surface, so every line is direct-labelled and has a table view.
- **Eigenvalues:** emphasis. Signal eigenvalues in slot 1, noise in grey.

Every chart carries native SVG <title> tooltips; the HTML report pairs each
chart with a table view so no value depends on hovering.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

# Categorical slots 1-3 (light, dark), validated all-pairs in both modes.
SERIES = (("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"))

# Diverging poles and midpoint (light, dark). Blue 600 and the project's "bad"
# red are of similar lightness, so neither arm looks heavier at equal |r|.
DIVERGING_NEGATIVE = ("#184f95", "#5598e7")
DIVERGING_POSITIVE = ("#b3261e", "#e66767")
DIVERGING_MID = ("#f0efec", "#383835")

# Heatmap bins: 20 bins of width 0.1 across [-1, 1].
BIN_WIDTH = 0.1
N_BINS = 20


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------
# Colour maths (OKLab), for perceptually even diverging steps
# --------------------------------------------------------------------------


def _hex_to_linear(hex_color: str) -> np.ndarray:
    rgb = np.array([int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in np.clip(srgb, 0, 1))


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _to_oklab(hex_color: str) -> np.ndarray:
    return _M2 @ np.cbrt(_M1 @ _hex_to_linear(hex_color))


def _from_oklab(lab: np.ndarray) -> str:
    lms = np.linalg.solve(_M2, lab) ** 3
    return _linear_to_hex(np.linalg.solve(_M1, lms))


def _mix(a: str, b: str, t: float) -> str:
    return _from_oklab(_to_oklab(a) * (1 - t) + _to_oklab(b) * t)


def diverging_steps(mode: int) -> list[str]:
    """Fill for each of the N_BINS bins, from -1 (bin 0) to +1 (last bin)."""
    mid, neg, pos = DIVERGING_MID[mode], DIVERGING_NEGATIVE[mode], DIVERGING_POSITIVE[mode]
    half = N_BINS // 2
    steps = []
    for i in range(N_BINS):
        centre = -1 + (i + 0.5) * BIN_WIDTH
        pole = neg if centre < 0 else pos
        steps.append(_mix(mid, pole, min(1.0, abs(centre) / (half * BIN_WIDTH) * 1.0)))
    return steps


def correlation_bin(value: float) -> int | None:
    if value is None or not math.isfinite(value):
        return None
    return int(min(N_BINS - 1, max(0, math.floor((value + 1) / BIN_WIDTH))))


def chart_css() -> str:
    """Custom properties and classes for every chart, in light and dark."""
    light, dark = diverging_steps(0), diverging_steps(1)

    def block(steps, series) -> str:
        lines = [f"--h{i}: {c};" for i, c in enumerate(steps)]
        lines += [f"--s{i + 1}: {c};" for i, c in enumerate(series)]
        return " ".join(lines)

    light_vars = block(light, [s[0] for s in SERIES])
    dark_vars = block(dark, [s[1] for s in SERIES])
    classes = " ".join(f".h{i} {{ fill: var(--h{i}); }}" for i in range(N_BINS))
    return f"""
:root {{ {light_vars} --cell-na: #e1e0d9; --ink-axis: #c3c2b7; --ink-grid: #e1e0d9;
  --ink-muted: #898781; --noise: #b5b3ab; }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{ {dark_vars} --cell-na: #2c2c2a;
    --ink-axis: #383835; --ink-grid: #2c2c2a; --noise: #5d5c57; }}
}}
:root[data-theme="dark"] {{ {dark_vars} --cell-na: #2c2c2a; --ink-axis: #383835;
  --ink-grid: #2c2c2a; --noise: #5d5c57; }}
{classes}
.hna {{ fill: var(--cell-na); }}
.c-axis {{ stroke: var(--ink-axis); stroke-width: 1; fill: none; }}
.c-grid {{ stroke: var(--ink-grid); stroke-width: 1; fill: none; }}
.c-zero {{ stroke: var(--ink-axis); stroke-width: 1.5; fill: none; }}
.c-label {{ font-size: 10px; fill: var(--ink-muted); }}
.c-label-strong {{ font-size: 11px; font-weight: 600; fill: var(--ink); }}
.c-tree {{ stroke: var(--ink-muted); stroke-width: 1.25; fill: none; }}
.c-line {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.c-s1 {{ stroke: var(--s1); }} .c-s2 {{ stroke: var(--s2); }} .c-s3 {{ stroke: var(--s3); }}
.f-s1 {{ fill: var(--s1); }} .f-s2 {{ fill: var(--s2); }} .f-s3 {{ fill: var(--s3); }}
.c-dot {{ fill-opacity: 0.45; }}
.c-noise {{ fill: var(--noise); }}
.c-threshold {{ stroke: var(--ink-muted); stroke-width: 1; stroke-dasharray: 4 3; }}
.c-hover {{ fill: transparent; }}
.c-hover:hover {{ fill: var(--ink-grid); fill-opacity: 0.5; }}
.heat-cell:hover {{ stroke: var(--ink); stroke-width: 1.5; }}
"""


# --------------------------------------------------------------------------
# Heatmap and its legend
# --------------------------------------------------------------------------


def heatmap(
    matrix: pd.DataFrame,
    order: Sequence[str],
    title: str,
    cell: int = 14,
    show_labels: bool = True,
    caption_value: str = "correlation",
) -> str:
    """Correlation heatmap, rows and columns in `order` (the cluster order)."""
    names = [n for n in order if n in matrix.index]
    n = len(names)
    label_space = 46 if show_labels else 4
    size = label_space + n * cell
    parts = [
        f'<svg viewBox="0 0 {size + 4} {size + 22}" class="heatmap" role="img" '
        f'aria-label="{_esc(title)}">',
        f'<text x="{label_space}" y="12" class="c-label-strong">{_esc(title)}</text>',
    ]
    top = 18 + label_space
    for i, row in enumerate(names):
        y = top + i * cell
        if show_labels:
            parts.append(
                f'<text x="{label_space - 4}" y="{y + cell * 0.72:.1f}" class="c-label" '
                f'text-anchor="end">{_esc(row)}</text>'
            )
            x = label_space + i * cell + cell * 0.7
            parts.append(
                f'<text x="{x:.1f}" y="{top - 4}" class="c-label" '
                f'transform="rotate(-60 {x:.1f} {top - 4})">{_esc(row)}</text>'
            )
        for j, col in enumerate(names):
            value = float(matrix.loc[row, col])
            b = correlation_bin(value)
            klass = f"h{b}" if b is not None else "hna"
            shown = "n/a" if b is None else f"{value:+.2f}"
            parts.append(
                f'<rect x="{label_space + j * cell}" y="{y}" width="{cell - 1}" '
                f'height="{cell - 1}" class="{klass} heat-cell"><title>{_esc(row)} / '
                f'{_esc(col)}: {caption_value} {shown}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def heatmap_legend() -> str:
    """Diverging scale legend: -1 ... 0 ... +1."""
    width, height = 260, 34
    step = width / N_BINS
    parts = [f'<svg viewBox="0 0 {width + 20} {height}" class="legend" role="img" '
             f'aria-label="Colour scale from -1 to +1">']
    for i in range(N_BINS):
        parts.append(f'<rect x="{10 + i * step:.1f}" y="4" width="{step - 1:.1f}" height="10" '
                     f'class="h{i}"/>')
    for value, x in ((-1, 10), (0, 10 + width / 2), (1, 10 + width)):
        parts.append(f'<text x="{x:.1f}" y="28" class="c-label" text-anchor="middle">'
                     f'{value:+d}</text>' if value else
                     f'<text x="{x:.1f}" y="28" class="c-label" text-anchor="middle">0</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Dendrogram
# --------------------------------------------------------------------------


def dendrogram(linkage: np.ndarray, labels: Sequence[str], order: Sequence[int]) -> str:
    """Horizontal dendrogram; leaves top to bottom in `order`, distance left to right."""
    n = len(labels)
    row = 14
    label_space = 46
    width = 520
    height = n * row + 30
    max_d = float(linkage[:, 2].max()) if len(linkage) else 1.0
    max_d = max(max_d, 1e-9)

    def xpos(d: float) -> float:
        return label_space + 6 + d / max_d * (width - label_space - 16)

    y_of: dict[int, float] = {leaf: 10 + i * row + row / 2 for i, leaf in enumerate(order)}
    d_of: dict[int, float] = {leaf: 0.0 for leaf in range(n)}
    parts = [f'<svg viewBox="0 0 {width} {height}" class="dendrogram" role="img" '
             f'aria-label="Hierarchical clustering of assets by correlation distance">']
    for i, leaf in enumerate(order):
        parts.append(f'<text x="{label_space}" y="{y_of[leaf] + 3.5:.1f}" class="c-label" '
                     f'text-anchor="end">{_esc(labels[leaf])}</text>')
    for k, (a, b, dist, size) in enumerate(linkage):
        a, b = int(a), int(b)
        node = n + k
        xa, xb, xm = xpos(d_of[a]), xpos(d_of[b]), xpos(dist)
        ya, yb = y_of[a], y_of[b]
        parts.append(
            f'<path d="M{xa:.1f},{ya:.1f} H{xm:.1f} V{yb:.1f} H{xb:.1f}" class="c-tree">'
            f'<title>merged at distance {dist:.3f} (correlation {1 - 2 * dist**2:+.2f}), '
            f'{int(size)} assets</title></path>'
        )
        y_of[node] = (ya + yb) / 2
        d_of[node] = dist
    axis_y = height - 16
    parts.append(f'<line x1="{xpos(0):.1f}" y1="{axis_y}" x2="{xpos(max_d):.1f}" y2="{axis_y}" '
                 f'class="c-axis"/>')
    for frac in (0, 0.5, 1):
        d = max_d * frac
        parts.append(f'<text x="{xpos(d):.1f}" y="{axis_y + 12}" class="c-label" '
                     f'text-anchor="middle">{d:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Lines over time
# --------------------------------------------------------------------------


def rolling_lines(series: pd.DataFrame, title: str, y_range=(-1.0, 1.0)) -> str:
    """Up to three rolling-correlation lines on one -1..+1 axis, with hover bands."""
    series = series.dropna(how="all")
    if series.empty:
        return ""
    width, height = 640, 240
    left, right, top, bottom = 34, 96, 14, 26
    plot_w, plot_h = width - left - right, height - top - bottom
    start, end = series.index.min(), series.index.max()
    span = max((end - start).days, 1)
    lo, hi = y_range

    def x(ts) -> float:
        return left + (ts - start).days / span * plot_w

    def y(v) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" class="lines" role="img" '
             f'aria-label="{_esc(title)}">']
    for v in (-1.0, -0.5, 0.5, 1.0):
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{left + plot_w}" y2="{y(v):.1f}" '
                     f'class="c-grid"/>')
        parts.append(f'<text x="{left - 4}" y="{y(v) + 3:.1f}" class="c-label" '
                     f'text-anchor="end">{v:+.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{left + plot_w}" y2="{y(0):.1f}" '
                 f'class="c-zero"/>')
    parts.append(f'<text x="{left - 4}" y="{y(0) + 3:.1f}" class="c-label" '
                 f'text-anchor="end">0</text>')
    for year in range(start.year + 1, end.year + 1):
        ts = pd.Timestamp(year=year, month=1, day=1)
        if (end.year - start.year) > 12 and year % 2:
            continue
        parts.append(f'<text x="{x(ts):.1f}" y="{height - 8}" class="c-label" '
                     f'text-anchor="middle">{year}</text>')

    labels_y: list[float] = []
    for i, column in enumerate(series.columns[:3]):
        s = series[column].dropna()
        if s.empty:
            continue
        # Weekly sampling keeps the path light without changing its shape.
        s = s.iloc[:: max(1, len(s) // 900)]
        points = " ".join(f"{x(ts):.1f},{y(v):.1f}" for ts, v in s.items())
        parts.append(f'<polyline points="{points}" class="c-line c-s{i + 1}"/>')
        ly = y(float(s.iloc[-1]))
        for prev in labels_y:  # keep end labels from overlapping
            if abs(ly - prev) < 12:
                ly = prev + 12 if ly >= prev else prev - 12
        labels_y.append(ly)
        parts.append(f'<text x="{left + plot_w + 6}" y="{ly + 3.5:.1f}" class="c-label-strong">'
                     f'{_esc(column)} {float(s.iloc[-1]):+.2f}</text>')

    # Hover bands: one per ~quarter, titled with every line's value there.
    edges = pd.date_range(start, end, freq="QS")
    edges = edges.insert(0, start).append(pd.DatetimeIndex([end])).unique()
    for a, b in zip(edges[:-1], edges[1:]):
        window = series.loc[a:b]
        if window.empty:
            continue
        values = ", ".join(
            f"{c} {window[c].dropna().iloc[-1]:+.2f}" for c in series.columns[:3]
            if not window[c].dropna().empty
        )
        parts.append(
            f'<rect x="{x(a):.1f}" y="{top}" width="{max(x(b) - x(a), 1):.1f}" '
            f'height="{plot_h}" class="c-hover"><title>{b.date()}: {_esc(values)}</title></rect>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Scatter small multiples
# --------------------------------------------------------------------------


def scatter(x: pd.Series, y: pd.Series, slot: int, title: str, note: str) -> str:
    """One scatter panel of paired returns, symmetric axes through zero."""
    frame = pd.concat([x, y], axis=1, sort=True).dropna()
    size, pad = 260, 30
    plot = size - 2 * pad
    limit = float(np.nanpercentile(np.abs(frame.to_numpy()), 99.5)) or 1e-6

    def pos(v: float) -> float:
        return pad + (max(-limit, min(limit, v)) + limit) / (2 * limit) * plot

    parts = [f'<svg viewBox="0 0 {size} {size + 14}" class="scatter" role="img" '
             f'aria-label="{_esc(title)}">',
             f'<text x="{pad}" y="12" class="c-label-strong">{_esc(title)}</text>',
             f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" class="c-grid"/>',
             f'<line x1="{pos(0):.1f}" y1="{pad}" x2="{pos(0):.1f}" y2="{pad + plot}" '
             f'class="c-axis"/>',
             f'<line x1="{pad}" y1="{size - pos(0):.1f}" x2="{pad + plot}" '
             f'y2="{size - pos(0):.1f}" class="c-axis"/>']
    for a, b in frame.to_numpy():
        parts.append(f'<circle cx="{pos(a):.1f}" cy="{size - pos(b):.1f}" r="1.8" '
                     f'class="f-s{slot} c-dot"/>')
    parts.append(f'<text x="{pad}" y="{size + 8}" class="c-label">{_esc(note)}</text>')
    parts.append(f'<text x="{pad + plot}" y="{size - pos(0) - 4:.1f}" class="c-label" '
                 f'text-anchor="end">{_esc(x.name)}</text>')
    parts.append(f'<text x="{pos(0) + 4:.1f}" y="{pad + 9}" class="c-label">{_esc(y.name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Eigenvalue spectrum
# --------------------------------------------------------------------------


def eigenvalue_bars(
    eigenvalues: np.ndarray, edge: float, lambda_plus: float, n_signal: int
) -> str:
    """Sorted eigenvalues on a log scale, signal vs noise, with the noise edge."""
    values = np.asarray(eigenvalues, dtype=float)
    n = len(values)
    width, height = 640, 220
    left, right, top, bottom = 40, 140, 12, 22
    plot_w, plot_h = width - left - right, height - top - bottom
    floor = max(values.min(), 1e-3)
    lo, hi = math.log10(floor) - 0.1, math.log10(values.max()) + 0.1

    def y(v: float) -> float:
        return top + (hi - math.log10(max(v, floor))) / (hi - lo) * plot_h

    gap = 2
    bar = plot_w / n
    parts = [f'<svg viewBox="0 0 {width} {height}" class="spectrum" role="img" '
             f'aria-label="Eigenvalues of the correlation matrix against the noise edge">']
    for tick in (0.01, 0.1, 1, 10, 100):
        if lo <= math.log10(tick) <= hi:
            parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" '
                         f'y2="{y(tick):.1f}" class="c-grid"/>')
            parts.append(f'<text x="{left - 4}" y="{y(tick) + 3:.1f}" class="c-label" '
                         f'text-anchor="end">{tick:g}</text>')
    base = top + plot_h
    for i, v in enumerate(values):
        klass = "f-s1" if i < n_signal else "c-noise"
        yy = y(v)
        parts.append(
            f'<rect x="{left + i * bar + gap / 2:.1f}" y="{yy:.1f}" '
            f'width="{max(bar - gap, 1):.1f}" height="{max(base - yy, 0.5):.1f}" rx="2" '
            f'class="{klass}"><title>Eigenvalue {i + 1}: {v:.3f} '
            f'({"signal" if i < n_signal else "noise"})</title></rect>'
        )
    thresholds = [(edge, "noise edge"), (lambda_plus, "MP bound")]
    if abs(y(edge) - y(lambda_plus)) < 12:  # too close to label separately
        thresholds = [(edge, "noise edge")]
    tip = (f"Noise edge (Tracy-Widom 95%) {edge:.3f}; "
           f"Marchenko-Pastur upper bound {lambda_plus:.3f}")
    for value, label in thresholds:
        if lo <= math.log10(max(value, floor)) <= hi:
            parts.append(f'<line x1="{left}" y1="{y(value):.1f}" x2="{left + plot_w}" '
                         f'y2="{y(value):.1f}" class="c-threshold"><title>{tip}</title></line>')
            parts.append(f'<text x="{left + plot_w + 6}" y="{y(value) + 3:.1f}" '
                         f'class="c-label">{label} {value:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Bars around zero (relative to a benchmark)
# --------------------------------------------------------------------------

# Symmetric axis limits for relative changes, as fractions.
_RELATIVE_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def _bar_path(x0: float, x1: float, y: float, height: float, radius: float = 4.0) -> str:
    """Horizontal bar from baseline x0 to data end x1: square at x0, rounded at x1."""
    r = min(radius, abs(x1 - x0), height / 2)
    if r <= 0:
        return f"M{x0:.1f},{y:.1f}h0.5v{height:.1f}h-0.5Z"
    d = 1 if x1 >= x0 else -1
    sweep = 1 if d > 0 else 0
    return (f"M{x0:.1f},{y:.1f}H{x1 - d * r:.1f}"
            f"A{r},{r} 0 0 {sweep} {x1:.1f},{y + r:.1f}V{y + height - r:.1f}"
            f"A{r},{r} 0 0 {sweep} {x1 - d * r:.1f},{y + height:.1f}H{x0:.1f}Z")


def relative_bars(rows: Sequence[tuple[str, float, bool, str]], title: str) -> str:
    """One bar per row: change against a benchmark, negative (better) to the left.

    rows: (label, relative change as a fraction, significant, tooltip). Colour
    is emphasis, as in the eigenvalue chart: slot 1 for a difference that
    survives the significance test, grey for one that does not. The direction
    and the direct label carry the sign, so nothing depends on colour alone.
    """
    rows = [r for r in rows if r[1] is not None and math.isfinite(r[1])]
    if not rows:
        return ""
    largest = max(abs(r[1]) for r in rows) * 1.35  # head-room for the value labels
    limit = next((s for s in _RELATIVE_STEPS if s >= largest), _RELATIVE_STEPS[-1])
    row_h, bar_h, label_w, width, right = 26, 14, 250, 640, 16
    plot_w = width - label_w - right
    top, axis_h = 6, 34
    height = top + len(rows) * row_h + axis_h

    def x(v: float) -> float:
        v = max(-limit, min(limit, v))
        return label_w + (v + limit) / (2 * limit) * plot_w

    zero = x(0.0)
    plot_bottom = top + len(rows) * row_h
    parts = [f'<svg viewBox="0 0 {width} {height}" class="bars" role="img" '
             f'aria-label="{_esc(title)}">']
    for tick in (-limit, -limit / 2, limit / 2, limit):
        parts.append(f'<line x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" '
                     f'y2="{plot_bottom}" class="c-grid"/>')
    for tick in (-limit, -limit / 2, 0.0, limit / 2, limit):
        parts.append(f'<text x="{x(tick):.1f}" y="{plot_bottom + 13}" class="c-label" '
                     f'text-anchor="middle">{tick:+.0%}</text>' if tick else
                     f'<text x="{zero:.1f}" y="{plot_bottom + 13}" class="c-label" '
                     f'text-anchor="middle">0</text>')
    parts.append(f'<text x="{zero - 6:.1f}" y="{plot_bottom + 28}" class="c-label" '
                 f'text-anchor="end">&#8592; lower than the benchmark (better)</text>')
    parts.append(f'<text x="{zero + 6:.1f}" y="{plot_bottom + 28}" class="c-label">'
                 f'higher (worse) &#8594;</text>')

    for i, (label, value, significant, tip) in enumerate(rows):
        y = top + i * row_h
        bar_y = y + (row_h - bar_h) / 2
        tip_x = x(value)
        klass = "f-s1" if significant else "c-noise"
        parts.append(f'<text x="{label_w - 10}" y="{y + 17}" class="c-label" '
                     f'text-anchor="end">{_esc(label)}</text>')
        parts.append(f'<path d="{_bar_path(zero, tip_x, bar_y, bar_h)}" class="{klass}"/>')
        anchor, offset = ("end", -5) if value < 0 else ("start", 5)
        parts.append(f'<text x="{tip_x + offset:.1f}" y="{y + 17}" class="c-label-strong" '
                     f'text-anchor="{anchor}">{value:+.1%}</text>')
        parts.append(f'<rect x="0" y="{y}" width="{width}" height="{row_h}" class="c-hover">'
                     f'<title>{_esc(tip)}</title></rect>')
    parts.append(f'<line x1="{zero:.1f}" y1="{top - 2}" x2="{zero:.1f}" y2="{plot_bottom}" '
                 f'class="c-zero"/>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Horizontal bars (one series)
# --------------------------------------------------------------------------


def value_bars(rows: Sequence[tuple[str, float, str]], value_range=(0.0, 1.0)) -> str:
    """Horizontal bars for one series: (label, value, detail)."""
    rows = [r for r in rows if r[1] is not None and math.isfinite(r[1])]
    if not rows:
        return ""
    row_h, label_w, width = 22, 210, 640
    plot_w = width - label_w - 80
    lo, hi = value_range
    height = len(rows) * row_h + 10

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" class="bars" role="img" '
             f'aria-label="Average correlation by regime">']
    parts.append(f'<line x1="{x(lo):.1f}" y1="0" x2="{x(lo):.1f}" y2="{height - 6}" '
                 f'class="c-axis"/>')
    for i, (label, value, detail) in enumerate(rows):
        yy = 4 + i * row_h
        parts.append(f'<text x="{label_w - 8}" y="{yy + 13}" class="c-label" '
                     f'text-anchor="end">{_esc(label)}</text>')
        parts.append(f'<rect x="{x(lo):.1f}" y="{yy + 3}" width="{max(x(value) - x(lo), 1):.1f}" '
                     f'height="{row_h - 8}" rx="2" class="f-s1"><title>{_esc(label)}: '
                     f'{value:+.2f} ({_esc(detail)})</title></rect>')
        parts.append(f'<text x="{x(value) + 6:.1f}" y="{yy + 13}" class="c-label-strong">'
                     f'{value:+.2f}</text>')
        parts.append(f'<text x="{x(value) + 44:.1f}" y="{yy + 13}" class="c-label">'
                     f'{_esc(detail)}</text>')
    parts.append("</svg>")
    return "".join(parts)
