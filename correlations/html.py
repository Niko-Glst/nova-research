"""Self-contained HTML report for a correlation analysis, plus its archive.

Kept apart from the single-ticker research reports, as the project owner
asked: correlation reports live in correlation_reports/ with their own index.
Each run writes

- `correlation_reports/<universe>.html`            the latest run, overwritten
- `correlation_reports/archive/<universe>_<base>_<stamp>.html`  never overwritten
- `correlation_reports/index.html`                 every archived run

The index is rebuilt from a small JSON block embedded in each archived report,
so it cannot drift from what is on disk.

Inline CSS and SVG only: no CDN, no network calls at view time. All external
text (tickers, labels) is escaped.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from correlations import charts
from correlations.analysis import CorrelationAnalysis, CurrencyView, FrequencyResult
from correlations.estimators import COVARIANCE_WINDOW, DEFAULT_COVARIANCE_METHOD
from correlations.evaluation import (
    FORECAST_ERROR,
    LONG_ONLY,
    UNCONSTRAINED,
    EstimatorScore,
    MetricRanking,
)
from correlations.anchoring import ANCHOR_WINDOW
from correlations.findings import build_findings, primary_view
from correlations.pair import PairAnalysis, summary_lines
from correlations.report import (
    LIMITATIONS,
    STRUCTURED_LABELS,
    average_correlation,
    evaluation_value,
    mean_abs_offdiagonal_gap,
    verdict,
)

REPORT_DIR = Path("correlation_reports")
ARCHIVE_DIR = "archive"
INDEX_FILE = "index.html"
_STAMP = "%Y-%m-%d_%H%M"
_META = re.compile(r'<script type="application/json" id="run-meta">(.*?)</script>', re.S)

# Pairs drawn over time and as scatters when present: equity against
# treasuries, gold and emerging markets, the classic diversification questions.
KEY_PAIRS = (("SPY", "TLT"), ("SPY", "GLD"), ("SPY", "EEM"))

# Rolling window drawn in the report (one trading year).
CHART_WINDOW = 252


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt(value: float | None, spec: str = "+.2f") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec)


PAGE_CSS = """
:root { --bg: #f9f9f7; --card: #fcfcfb; --ink: #0b0b0b; --muted: #52514e;
  --line: #e1e0d9; --risk: #b3261e; --risk-bg: #fbeae9; --good: #006300;
  --good-bg: #e6f4ec; --neutral-bg: #f0efec; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) { --bg: #0d0d0d; --card: #1a1a19; --ink: #ffffff;
    --muted: #c3c2b7; --line: #2c2c2a; --risk: #e66767; --risk-bg: #2e1a19;
    --good: #0ca30c; --good-bg: #142414; --neutral-bg: #242422; color-scheme: dark; }
}
:root[data-theme="dark"] { --bg: #0d0d0d; --card: #1a1a19; --ink: #ffffff; --muted: #c3c2b7;
  --line: #2c2c2a; --risk: #e66767; --risk-bg: #2e1a19; --good: #0ca30c; --good-bg: #142414;
  --neutral-bg: #242422; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.4px; }
h2 { font-size: 18px; margin: 0 0 6px; }
h3 { font-size: 13px; margin: 18px 0 6px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--muted); }
.brand { font-size: 10px; font-weight: 700; letter-spacing: 2.2px; color: var(--muted); }
.muted { color: var(--muted); font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px;
  margin: 16px 0; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .value { font-size: 26px; font-weight: 650; margin-top: 2px; }
.kpi .detail { font-size: 12px; color: var(--muted); }
.findings { display: grid; gap: 10px; }
.finding { border-left: 3px solid var(--line); padding: 8px 12px; border-radius: 4px;
  background: var(--neutral-bg); }
.finding.risk { border-left-color: var(--risk); background: var(--risk-bg); }
.finding.good { border-left-color: var(--good); background: var(--good-bg); }
.finding b { display: block; margin-bottom: 2px; }
.grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
svg { width: 100%; height: auto; display: block; }
.legend-wrap { max-width: 320px; margin: 4px 0 0 46px; }
.dendro-wrap { max-width: 640px; }
.single-map { max-width: 500px; }
.scatter-wrap svg { max-width: 320px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
.scroll { overflow-x: auto; }
.matrix td, .matrix th { padding: 3px 5px; font-size: 11px; }
.legend-row svg { display: inline-block; width: 12px; height: 12px; vertical-align: -1px; }
.bars-wrap { max-width: 700px; }
ul { margin: 6px 0 0; padding-left: 18px; }
li { margin-bottom: 4px; }
footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 24px; }
@media (max-width: 620px) { .grid2, .grid3 { grid-template-columns: 1fr; } }
"""


# --------------------------------------------------------------------------
# Small builders
# --------------------------------------------------------------------------


def _kpi(label: str, value: str, detail: str) -> str:
    return (f'<div class="kpi"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div><div class="detail">{_esc(detail)}</div></div>')


def _matrix_table(matrix: pd.DataFrame, order: list[str]) -> str:
    names = [n for n in order if n in matrix.index]
    head = "".join(f'<th class="num">{_esc(n)}</th>' for n in names)
    rows = []
    for r in names:
        cells = "".join(f'<td class="num">{_fmt(float(matrix.loc[r, c]))}</td>' for c in names)
        rows.append(f"<tr><th>{_esc(r)}</th>{cells}</tr>")
    return (f'<details><summary>Table view</summary><div class="scroll"><table class="matrix">'
            f"<tr><th></th>{head}</tr>{''.join(rows)}</table></div></details>")


def _key_pairs(result: FrequencyResult) -> list[tuple[str, str]]:
    assets = set(result.returns.columns)
    chosen = [p for p in KEY_PAIRS if p[0] in assets and p[1] in assets]
    if len(chosen) < 3:
        longest = next((r for r in reversed(result.rolling) if r.paths), None)
        extra = sorted(longest.paths, key=lambda q: -q.range) if longest else []
        for q in extra:
            pair = (q.asset_a, q.asset_b)
            if pair not in chosen:
                chosen.append(pair)
            if len(chosen) == 3:
                break
    return chosen[:3]


def _rolling_frame(result: FrequencyResult, pairs, window: int) -> pd.DataFrame:
    rolling = next((r for r in result.rolling if r.window == window), None)
    if rolling is None or rolling.series.empty:
        return pd.DataFrame()
    columns = {}
    for a, b in pairs:
        for key in (f"{a}|{b}", f"{b}|{a}"):
            if key in rolling.series:
                columns[f"{a}-{b}"] = rolling.series[key]
                break
    return pd.DataFrame(columns)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _matrices_section(result: FrequencyResult, order: list[str]) -> str:
    est = result.estimates
    sample = est["pearson"]
    shown = [("pearson", "Sample (Pearson)"), ("denoised", "Random-matrix denoised")]
    maps = "".join(
        f"<div>{charts.heatmap(est[k].matrix, order, label)}</div>"
        for k, label in shown if k in est and not est[k].matrix.isna().all().all()
    )
    lw = est.get("ledoit_wolf")
    lw_note = ""
    if lw is not None and "intensity" in lw.parameters:
        gap = mean_abs_offdiagonal_gap(lw.matrix, sample.matrix)
        lw_note = (f"<p class='muted'>Ledoit-Wolf shrinkage is not drawn: on this much data it "
                   f"puts a weight of only {lw.parameters['intensity']:.3f} on its target and "
                   f"moves the average pair by {gap:.3f}, so its map is indistinguishable from "
                   f"the sample. Shrinkage matters on short estimation windows: the "
                   f"out-of-sample section below tests it on one-year windows.</p>")
    return f"""
    <section class="card">
      <h2>Correlation matrices</h2>
      <p class="muted">Daily log returns, assets ordered by the clustering below so related
      assets sit together. Red is positive, blue negative. Hover a cell for its value.</p>
      <div class="grid2">{maps}</div>
      <div class="legend-wrap">{charts.heatmap_legend()}</div>
      {lw_note}
      {_matrix_table(sample.matrix, order)}
    </section>"""


def _estimator_table(result: FrequencyResult) -> str:
    sample = result.estimates["pearson"]
    rows = []
    for key, label in (("pearson", "Sample Pearson"), ("spearman", "Spearman (rank)"),
                       ("kendall", "Kendall tau-b"), ("ewma", "EWMA (RiskMetrics)"),
                       *STRUCTURED_LABELS.items()):
        est = result.estimates.get(key)
        if est is None:
            continue
        if est.matrix.isna().all().all():
            rows.append(f"<tr><td>{_esc(label)}</td><td colspan=4>not estimable: "
                        f"{_esc(est.warnings[-1] if est.warnings else '')}</td></tr>")
            continue
        params = ", ".join(f"{k} {v}" for k, v in est.parameters.items()
                           if k not in ("demeaned",))
        gap = mean_abs_offdiagonal_gap(est.matrix, sample.matrix)
        # Kendall's tau lives on a different scale (about (2/pi) asin r), so its
        # distance from Pearson is not an estimation difference.
        if key == "pearson":
            gap_text = ""
        elif key == "kendall":
            gap_text = "other scale"
        else:
            gap_text = _fmt(gap, ".3f")
        rows.append(
            f"<tr><td>{_esc(label)}</td><td class='num'>{_fmt(average_correlation(est.matrix))}"
            f"</td><td class='num'>{gap_text}</td>"
            f"<td>{'yes' if est.is_psd else 'NO'}</td><td class='muted'>{_esc(params)}</td></tr>"
        )
    return ("<table><tr><th>Estimator</th><th class='num'>Avg corr</th>"
            "<th class='num'>Avg |change| vs sample</th><th>PSD</th><th>Parameters</th></tr>"
            + "".join(rows) + "</table>")


def _spectrum_section(result: FrequencyResult) -> str:
    est = result.estimates.get("denoised")
    if est is None or est.matrix.isna().all().all():
        return ""
    values = np.sort(np.linalg.eigvalsh(result.estimates["pearson"].matrix.to_numpy()))[::-1]
    p = est.parameters
    chart = charts.eigenvalue_bars(values, float(p["signal_edge"]), float(p["lambda_plus"]),
                                   int(p["n_signal"]))
    warn = "".join(f"<li>{_esc(w)}</li>" for w in est.warnings)
    return f"""
    <section class="card">
      <h2>Signal and noise</h2>
      <p class="muted">Eigenvalues of the sample correlation matrix, largest first, on a log
      scale. Blue bars clear the finite-sample noise edge and are treated as structure; grey
      ones are indistinguishable from what pure noise would produce with T/N = {p['q']}.
      Denoising keeps the blue ones and flattens the grey ones.</p>
      {chart}
      <h3>All estimators</h3>
      {_estimator_table(result)}
      {f'<ul class="muted">{warn}</ul>' if warn else ''}
    </section>"""


def _regime_section(result: FrequencyResult, order: list[str]) -> str:
    r = result.regimes
    if r is None:
        return ""
    calm, high = r.regime("vix_low"), r.regime("vix_high")
    maps = ""
    if calm is not None and high is not None and not high.estimate.matrix.isna().all().all():
        diff = high.estimate.matrix - calm.estimate.matrix
        maps = (
            f'<div class="grid2"><div>{charts.heatmap(calm.estimate.matrix, order, calm.label)}'
            f'</div><div>{charts.heatmap(high.estimate.matrix, order, high.label)}</div>'
            f'<div>{charts.heatmap(diff, order, "Change (stressed minus calm)", caption_value="change")}'
            f'</div></div><div class="legend-wrap">{charts.heatmap_legend()}</div>'
        )
    bars = []
    table = []
    for g in r.regimes:
        avg = average_correlation(g.estimate.matrix) if not g.estimate.matrix.isna().all().all() \
            else float("nan")
        noisy = " (noisy: below 60)" if 0 < g.n_observations < 60 and math.isfinite(avg) else ""
        bars.append((g.label, avg, f"n {g.n_observations}{noisy}"))
        note = "; ".join(g.warnings[-1:]) if not math.isfinite(avg) else noisy.strip(" ()")
        table.append(f"<tr><td>{_esc(g.label)}</td><td class='muted'>{_esc(g.definition)}</td>"
                     f"<td class='num'>{g.n_observations}</td><td class='num'>{_fmt(avg)}</td>"
                     f"<td class='muted'>{_esc(note)}</td></tr>")
    full = average_correlation(result.estimates["pearson"].matrix)
    bars.insert(0, ("Full sample", full, f"n {result.estimates['pearson'].n_observations}"))
    shifts = "".join(
        f"<tr><td>{_esc(s.asset_a)}-{_esc(s.asset_b)}</td><td class='num'>{_fmt(s.calm)}</td>"
        f"<td class='num'>{_fmt(s.stressed)}</td><td class='num'>{_fmt(s.change)}</td></tr>"
        for s in r.shifts[:15]
    )
    return f"""
    <section class="card">
      <h2>Calm and stressed markets</h2>
      <p class="muted">Correlation on days with the VIX at or below 30 against days above 30, and
      the change between them. Red in the third map means the pair moves together more under
      stress, exactly when diversification is needed.</p>
      {maps}
      <h3>Average pairwise correlation by regime</h3>
      {charts.value_bars(bars, (min(0.0, min((b[1] for b in bars if math.isfinite(b[1])), default=0)), 1.0))}
      <details><summary>Table view and definitions</summary><table>
        <tr><th>Regime</th><th>Definition</th><th class="num">n</th><th class="num">Avg corr</th>
        <th>Note</th></tr>{''.join(table)}</table></details>
      <h3>Largest losses of diversification (VIX above 30)</h3>
      {f'<table><tr><th>Pair</th><th class="num">Calm</th><th class="num">Stressed</th><th class="num">Change</th></tr>{shifts}</table>' if shifts else '<p class="muted">No pair rises by 0.20 or more.</p>'}
    </section>"""


def _tail_section(result: FrequencyResult) -> str:
    r = result.regimes
    if r is None:
        return ""
    asym = sorted((a for a in r.asymmetry if math.isfinite(a.difference)),
                  key=lambda a: (not a.significant_adjusted, -a.difference))[:12]
    n_sig = sum(a.significant_adjusted for a in r.asymmetry)
    tails = sorted((t for t in r.tails if t.quantile == 0.05), key=lambda t: -t.lower_excess)[:12]
    down_rows = "".join(
        f"<tr><td>{_esc(a.asset_a)}-{_esc(a.asset_b)}</td><td class='num'>{_fmt(a.down)}</td>"
        f"<td class='num'>{_fmt(a.up)}</td><td class='num'>{_fmt(a.difference)}</td>"
        f"<td class='num'>{_fmt(a.p_adjusted, '.3f')}</td>"
        f"<td>{'yes' if a.significant_adjusted else 'no'}</td></tr>" for a in asym
    )
    tail_rows = "".join(
        f"<tr><td>{_esc(t.asset_a)}-{_esc(t.asset_b)}</td><td class='num'>{t.lower:.2f}</td>"
        f"<td class='num'>{t.lower_benchmark:.2f}</td><td class='num'>{t.lower_excess:+.2f}</td>"
        f"<td class='num'>{t.joint_lower_days}</td></tr>" for t in tails
    )
    return f"""
    <section class="card">
      <h2>Tail risk</h2>
      <p class="muted">Selecting down days changes a correlation even when nothing about the
      relationship changes (Forbes and Rigobon), and fat tails raise it further. The clean
      test compares each pair on days both fell with days both rose: any symmetric
      distribution gives the same value for both. Standard errors come from a block bootstrap,
      so fat tails and volatility clustering are allowed for; p-values are adjusted for the
      {len(r.asymmetry)} pairs tested. {n_sig} differ significantly.</p>
      <div class="grid2">
        <div><h3>Down days against up days</h3><table><tr><th>Pair</th>
          <th class="num">Both down</th><th class="num">Both up</th><th class="num">Gap</th>
          <th class="num">Adj. p</th><th>Signif.</th></tr>{down_rows}</table></div>
        <div><h3>Crashing together (worst 5% of days)</h3><table><tr><th>Pair</th>
          <th class="num">Observed</th><th class="num">Normal model</th><th class="num">Excess</th>
          <th class="num">Joint days</th></tr>{tail_rows}</table>
          <p class="muted">Share of an asset's worst-5% days on which the other was also in its
          worst 5%, scaled so independent assets score 0.05 and perfectly co-crashing ones 1.</p>
        </div>
      </div>
    </section>"""


def _over_time_section(result: FrequencyResult, weekly: FrequencyResult | None) -> str:
    pairs = _key_pairs(result)
    frame = _rolling_frame(result, pairs, CHART_WINDOW)
    if frame.empty:
        return ""
    lines = charts.rolling_lines(frame, f"Rolling {CHART_WINDOW}-day correlation")
    rows = []
    longest = next((r for r in result.rolling if r.window == CHART_WINDOW), None)
    for (a, b) in pairs:
        path = next((q for q in (longest.paths if longest else ())
                     if {q.asset_a, q.asset_b} == {a, b}), None)
        if path:
            rows.append(f"<tr><td>{_esc(a)}-{_esc(b)}</td><td class='num'>{_fmt(path.full_sample)}"
                        f"</td><td class='num'>{_fmt(path.minimum)}</td><td class='num'>"
                        f"{_fmt(path.maximum)}</td><td class='num'>{_fmt(path.latest)}</td></tr>")
    scatters = ""
    if weekly is not None:
        panels = []
        pearson_w = weekly.estimates["pearson"]
        spearman_w = weekly.estimates["spearman"]
        for slot, (a, b) in enumerate(pairs, start=1):
            if a in weekly.returns and b in weekly.returns:
                note = (f"Pearson {pearson_w.pair(a, b):+.2f}, Spearman {spearman_w.pair(a, b):+.2f},"
                        f" n {pearson_w.n_observations}")
                panels.append(charts.scatter(weekly.returns[a], weekly.returns[b], slot,
                                             f"{a} vs {b}", note))
        if panels:
            scatters = (f"<h3>Weekly returns, pair by pair</h3><p class='muted'>A scatter shows what a "
                        f"single coefficient cannot: whether the link is linear, driven by a few "
                        f"extreme weeks, or changes shape. Axes clipped at the 99.5th percentile."
                        f"</p><div class='grid3'>{''.join(panels)}</div>")
    return f"""
    <section class="card">
      <h2>How the key relationships moved</h2>
      <p class="muted">A correlation measured over one year at a time. The full-sample figure is
      an average over periods that can look nothing alike.</p>
      {lines}
      <details><summary>Table view</summary><table><tr><th>Pair</th><th class="num">Full sample</th>
      <th class="num">Lowest</th><th class="num">Highest</th><th class="num">Latest</th></tr>
      {''.join(rows)}</table></details>
      {scatters}
    </section>"""


def _score_tip(score: EstimatorScore, metric: str) -> str:
    return (f"{score.label}: {evaluation_value(metric, score.value)}, "
            f"{score.relative_to_benchmark:+.1%} vs benchmark, lower in {score.win_rate:.0%} of "
            f"months, adjusted p {_fmt(score.p_adjusted, '.3f')} ({verdict(score)})")


def _ranking_table(ranking: MetricRanking) -> str:
    metric = ranking.metric
    extra_head = {
        LONG_ONLY: "<th class='num'>Turnover</th><th class='num'>Eff. holdings</th>",
        UNCONSTRAINED: "<th class='num'>Gross exposure</th>",
    }.get(metric, "")
    rows = []
    for s in ranking.scores:
        extra = {
            LONG_ONLY: f"<td class='num'>{_fmt(s.turnover, '.3f')}</td>"
                       f"<td class='num'>{_fmt(s.effective_holdings, '.1f')}</td>",
            UNCONSTRAINED: f"<td class='num'>{_fmt(s.gross_exposure, '.2f')}</td>",
        }.get(metric, "")
        bench = s.is_benchmark
        rows.append(
            f"<tr><td class='num'>{s.rank}</td><td>{_esc(s.label)}</td>"
            f"<td class='num'>{_esc(evaluation_value(metric, s.value))}</td>"
            f"<td class='num'>{'' if bench else _fmt(s.relative_to_benchmark, '+.1%')}</td>"
            f"<td class='num'>{'' if bench else _fmt(s.win_rate, '.0%')}</td>"
            f"<td class='num'>{'' if bench else _fmt(s.p_adjusted, '.3f')}</td>"
            f"<td>{_esc(verdict(s))}</td>{extra}</tr>"
        )
    return (f"<h3>{_esc(ranking.label)}</h3><div class='scroll'><table><tr><th class='num'>#</th>"
            f"<th>Estimator</th><th class='num'>Value</th><th class='num'>vs benchmark</th>"
            f"<th class='num'>Months better</th><th class='num'>Adj. p</th><th>Verdict</th>"
            f"{extra_head}</tr>{''.join(rows)}</table></div>")


def _evaluation_section(analysis: CorrelationAnalysis) -> str:
    result = analysis.evaluation
    if result is None:
        return ""
    if not result.evaluated:
        reason = result.warnings[-1] if result.warnings else ""
        return (f"<section class='card'><h2>Which estimator forecasts best</h2>"
                f"<p class='muted'>Not evaluated: {_esc(reason)}</p></section>")
    charts_html = []
    for metric, title in ((FORECAST_ERROR, "Forecast error vs the 252-day sample"),
                          (LONG_ONLY, "Long-only minimum-variance volatility vs the sample")):
        ranking = result.ranking(metric)
        rows = [(s.label, s.relative_to_benchmark, s.significant_adjusted, _score_tip(s, metric))
                for s in ranking.scores if not s.is_benchmark]
        charts_html.append(f"<div class='bars-wrap'><h3>{_esc(title)}</h3>"
                           f"{charts.relative_bars(rows, title)}</div>")
    bench = next(c.label for c in result.contestants if c.key == result.benchmark)
    notes = [result.concentration_note()] + list(result.warnings)
    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in notes if n)
    legend = ("<p class='muted legend-row'><svg width='12' height='12' aria-hidden='true'>"
              "<rect width='12' height='12' rx='3' class='f-s1'/></svg> different from the "
              "benchmark after the multiple-comparison correction &nbsp; "
              "<svg width='12' height='12' aria-hidden='true'><rect width='12' height='12' "
              "rx='3' class='c-noise'/></svg> no significant difference</p>")
    return f"""
    <section class="card">
      <h2>Which estimator forecasts best (out of sample)</h2>
      <p class="muted">At the end of each month every estimator saw only the previous
      {result.estimation_window} trading days (the short samples fewer) and forecast the next
      month, which it had not seen: {result.n_periods} forecasts from
      {_esc(result.first_forecast.date())} to {_esc(result.last_forecast.date())}. Two scores: the
      distance to the correlation matrix that followed, and the volatility a minimum-variance
      portfolio built from the forecast actually had (every estimator combined with the same
      trailing volatilities, so only the correlation differs). Benchmark: {_esc(bench)}.
      Significance: Diebold-Mariano test on the monthly differences with a Newey-West variance,
      then Benjamini-Hochberg across estimators.</p>
      {legend}
      {''.join(charts_html)}
      {''.join(_ranking_table(r) for r in result.rankings)}
      <p class="muted">The allocation API, <code>estimate_covariance()</code>, uses
      {_esc(DEFAULT_COVARIANCE_METHOD)} on {COVARIANCE_WINDOW} days by default. Returns:
      {_esc(result.currency_convention)}</p>
      {f'<ul class="muted">{notes_html}</ul>' if notes_html else ''}
    </section>"""


def _currency_section(analysis: CorrelationAnalysis) -> str:
    local, base = analysis.view("local"), primary_view(analysis)
    if base is local or local is None:
        return ""
    rows = []
    checks = "".join(
        f"<li>{_esc(c.proxy)} vs {_esc(c.fx_symbol)}: "
        + ", ".join(f"lag {k:+d} {v:+.3f}" for k, v in c.correlations.items())
        + (" (OK)" if c.aligned and not c.note else f" ({_esc(c.note)})") + "</li>"
        for c in base.fx_timing
    )
    lw, bw = local.frequency("weekly"), base.frequency("weekly")
    if lw and bw:
        lm, bm = lw.estimates["pearson"].matrix, bw.estimates["pearson"].matrix
        gaps = []
        names = list(bm.columns)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                gaps.append((a, b, float(lm.loc[a, b]), float(bm.loc[a, b])))
        gaps.sort(key=lambda t: -abs(t[3] - t[2]))
        rows = [f"<tr><td>{_esc(a)}-{_esc(b)}</td><td class='num'>{_fmt(lv)}</td>"
                f"<td class='num'>{_fmt(bv)}</td><td class='num'>{_fmt(bv - lv)}</td></tr>"
                for a, b, lv, bv in gaps[:12]]
    return f"""
    <section class="card">
      <h2>Currency effect ({_esc(base.name)} investor)</h2>
      <p class="muted">{_esc(base.convention)}</p>
      <h3>Largest changes, weekly returns</h3>
      <table><tr><th>Pair</th><th class="num">Local currency</th><th class="num">{_esc(base.name)}</th>
      <th class="num">Change</th></tr>{''.join(rows)}</table>
      <h3>FX timing check</h3><ul class="muted">{checks}</ul>
    </section>"""


def _data_section(analysis: CorrelationAnalysis, view: CurrencyView) -> str:
    log = view.alignment
    missing = ", ".join(f"{_esc(s)} {len(d)}" for s, d in log.missing_dates.items() if d)
    epps = "".join(f"<li>{_esc(f.asset_a)}-{_esc(f.asset_b)}: daily {f.daily:+.2f}, weekly "
                   f"{f.weekly:+.2f} ({_esc(f.likely_cause)})</li>" for f in view.epps)
    daily = view.frequency("daily")
    outliers = "".join(
        f"<li>{_esc(o.symbol)} {o.date.date()}: {o.log_return:+.2%} (robust z {o.robust_z:+.1f})</li>"
        for o in (daily.outliers[:8] if daily else ())
    )
    fetch = "".join(f"<li>{_esc(e)}</li>" for e in analysis.fetch_log)
    notes = "".join(f"<li>{_esc(n)}</li>" for n in analysis.notes)
    limits = "".join(f"<li>{_esc(t)}</li>" for t in LIMITATIONS)
    return f"""
    <section class="card">
      <h2>Data and method</h2>
      <p class="muted">{_esc(view.convention)}</p>
      <ul class="muted">
        <li>Common calendar: {log.common_dates:,} dates, {log.common_start.date() if log.common_start is not None else 'n/a'}
          to {log.common_end.date() if log.common_end is not None else 'n/a'}; history limited by
          {_esc(log.limiting_symbol)}.</li>
        <li>Prices are aligned on dates every market traded before returns are taken, so a
          holiday becomes a multi-day return, never a zero. Dates dropped because a market was
          closed: {missing or 'none'}.</li>
        {notes}
      </ul>
      {f'<h3>Daily vs weekly divergence</h3><ul class="muted">{epps}</ul>' if epps else ''}
      {f'<h3>Largest outliers (flagged, not removed)</h3><ul class="muted">{outliers}</ul>' if outliers else ''}
      <details><summary>Data log</summary><ul class="muted">{fetch}</ul></details>
      <h3>Limitations</h3><ul class="muted">{limits}</ul>
    </section>"""


def _other_view_section(view: CurrencyView, order: list[str]) -> str:
    daily = view.frequency("daily") or view.frequency("weekly")
    if daily is None or daily.estimates["pearson"].matrix.isna().all().all():
        return ""
    title = "local currencies" if view.name == "local" else view.name
    return f"""
    <section class="card">
      <h2>The same universe in {_esc(title)}</h2>
      <p class="muted">{_esc(view.convention)}</p>
      <div class="single-map">{charts.heatmap(daily.estimates["pearson"].matrix, order,
      f"Sample, {daily.frequency}")}</div>
      {_matrix_table(daily.estimates["pearson"].matrix, order)}
    </section>"""


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def run_metadata(analysis: CorrelationAnalysis) -> dict:
    view = primary_view(analysis)
    daily = view.frequency("daily")
    meta = {
        "universe": analysis.universe.name,
        "base": analysis.base_currency,
        "view": view.name,
        "generated": analysis.generated_at.isoformat() if analysis.generated_at else "",
        "assets": len(analysis.universe.symbols),
    }
    if daily is not None:
        sample = daily.estimates["pearson"]
        meta.update(
            n=sample.n_observations,
            start=str(sample.start.date()) if sample.start is not None else "",
            end=str(sample.end.date()) if sample.end is not None else "",
            average_correlation=round(average_correlation(sample.matrix), 4),
            signal=daily.estimates["denoised"].parameters.get("n_signal"),
        )
        if daily.regimes is not None and daily.regimes.regime("vix_high") is not None:
            high = daily.regimes.regime("vix_high").estimate
            if not high.matrix.isna().all().all():
                meta["average_correlation_vix_high"] = round(average_correlation(high.matrix), 4)
            meta["pairs_losing_diversification"] = len(daily.regimes.shifts)
    result = analysis.evaluation
    if result is not None and result.evaluated:
        best = result.ranking(FORECAST_ERROR).winner
        meta["oos_best"] = best.label if best.better else "no clear winner"
        meta["oos_text"] = f"best forecaster: {meta['oos_best']}"
        meta["oos_periods"] = result.n_periods
    return meta


def render(analysis: CorrelationAnalysis) -> str:
    view = primary_view(analysis)
    daily = view.frequency("daily")
    weekly = view.frequency("weekly")
    lead = daily or weekly
    order = list(lead.clustering.ordered_labels) if lead and lead.clustering else list(
        analysis.universe.symbols)
    meta = run_metadata(analysis)

    kpis = ""
    if lead is not None:
        sample = lead.estimates["pearson"]
        kpis += _kpi("Average correlation", _fmt(average_correlation(sample.matrix)),
                     f"{lead.frequency}, n {sample.n_observations:,}")
        if "average_correlation_vix_high" in meta:
            kpis += _kpi("When the VIX is above 30", _fmt(meta["average_correlation_vix_high"]),
                         f"{meta['pairs_losing_diversification']} pairs lose 0.20 or more")
        if meta.get("signal") is not None:
            kpis += _kpi("Independent signals", str(meta["signal"]),
                         f"of {meta['assets']} assets (random matrix theory)")
        lw = lead.estimates.get("ledoit_wolf")
        if lw is not None and "intensity" in lw.parameters:
            kpis += _kpi("Shrinkage intensity", f"{lw.parameters['intensity']:.3f}",
                         "Ledoit-Wolf weight on the target")

    findings = "".join(
        f'<div class="finding {_esc(f.tone)}"><b>{_esc(f.title)}</b>{_esc(f.text)}</div>'
        for f in build_findings(analysis)
    )
    other = [v for v in analysis.views if v is not view]
    sections = ""
    if lead is not None:
        sections += _matrices_section(lead, order)
        if lead.clustering is not None:
            sections += f"""
    <section class="card"><h2>Clusters</h2>
      <p class="muted">Assets merged bottom-up by correlation distance sqrt((1 - r) / 2), average
      linkage. The further right two branches join, the less related they are.</p>
      <div class="dendro-wrap">{charts.dendrogram(lead.clustering.linkage,
      list(lead.clustering.labels), list(lead.clustering.order))}</div></section>"""
        sections += _regime_section(lead, order) if daily is not None else ""
        sections += _tail_section(lead) if daily is not None else ""
        sections += _over_time_section(lead, weekly)
        sections += _spectrum_section(lead)
    sections += _evaluation_section(analysis)
    sections += _currency_section(analysis)
    for v in other:
        sections += _other_view_section(v, order)
    sections += _data_section(analysis, view)

    stamp = analysis.generated_at.strftime("%Y-%m-%d %H:%M UTC") if analysis.generated_at else ""
    view_name = "local currencies" if view.name == "local" else f"{view.name} (base currency)"
    title = f"Correlation analysis: {analysis.universe.name}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} &middot; Nikolay Gelshtein</title>
<style>{PAGE_CSS}{charts.chart_css()}</style>
<script type="application/json" id="run-meta">{_esc(json.dumps(meta))}</script>
</head>
<body><div class="wrap">
<header>
  <div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
  <h1>{_esc(title)}</h1>
  <p class="muted">{_esc(len(analysis.universe.symbols))} assets, returns in {_esc(view_name)}.
  Generated {_esc(stamp)}. {_esc(analysis.universe.description)}</p>
</header>
<div class="kpis">{kpis}</div>
<section class="card"><h2>Findings</h2><div class="findings">{findings}</div></section>
{sections}
<footer>Research output. Not investment advice. No orders are placed.<br>
Figures are point-in-time reads from public data and may be stale or wrong.</footer>
</div></body></html>"""


# --------------------------------------------------------------------------
# Archive and index
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchivedRun:
    path: Path
    meta: dict


def _write(document: str, name: str, base: str, directory: Path | str,
           now: datetime | None) -> Path:
    """Latest copy, a dated archive copy that is never overwritten, and the index."""
    now = now or datetime.now()
    root = Path(directory)
    archive = root / ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    latest = root / f"{name}.html"
    latest.write_text(document, encoding="utf-8")
    (archive / f"{name}_{base}_{now.strftime(_STAMP)}.html").write_text(document, encoding="utf-8")
    write_index(root)
    return latest


def save(analysis: CorrelationAnalysis, directory: Path | str = REPORT_DIR,
         now: datetime | None = None) -> Path:
    """Write the latest copy, a dated archive copy, and rebuild the index."""
    return _write(render(analysis), analysis.universe.name, analysis.base_currency, directory, now)


def save_pair(pair: PairAnalysis, directory: Path | str = REPORT_DIR,
              now: datetime | None = None) -> Path:
    """Pair report: correlation_reports/pair_<A>_<B>.html, archived like any run."""
    return _write(render_pair(pair), f"pair_{pair.asset_a}_{pair.asset_b}", pair.base_currency,
                  directory, now)


# --------------------------------------------------------------------------
# Pair report
# --------------------------------------------------------------------------


def _figure_rows(figures) -> str:
    return "".join(
        f"<tr><td>{_esc(f.label)}</td><td class='num'>{_fmt(f.value)}</td>"
        f"<td class='num'>{f.n_observations:,}</td><td class='muted'>"
        f"{_esc(f.detail if len(f.detail) < 90 else '')}</td></tr>"
        for f in figures
    )


def pair_metadata(pair: PairAnalysis) -> dict:
    full = pair.figure("Full history, daily")
    twelve = pair.forecast(12)
    meta = {
        "universe": pair.name,
        "base": pair.base_currency,
        "view": pair.view,
        "generated": pair.generated_at.isoformat() if pair.generated_at else "",
        "assets": 2,
        "n": pair.n_days,
        "start": str(pair.start.date()),
        "end": str(pair.end.date()),
        "average_correlation": round(full.value, 4) if full else None,
    }
    if pair.stress is not None and math.isfinite(pair.stress.value):
        meta["average_correlation_vix_high"] = round(pair.stress.value, 4)
    if twelve is not None:
        meta["oos_text"] = f"12-month forecast {twelve.forecast:+.2f}"
    return meta


def render_pair(pair: PairAnalysis) -> str:
    meta = pair_metadata(pair)
    full, year = pair.figure("Full history, daily"), pair.figure("Last 12 months")
    twelve = pair.forecast(12)
    kpis = ""
    if full:
        kpis += _kpi("Full history", _fmt(full.value), f"daily, n {full.n_observations:,}")
    if year:
        kpis += _kpi("Last 12 months", _fmt(year.value), year.detail)
    if twelve:
        band = (f"80% range {_fmt(twelve.low)} to {_fmt(twelve.high)}"
                if math.isfinite(twelve.low) else "no range yet")
        kpis += _kpi("Forecast, next 12 months", _fmt(twelve.forecast), band)
    if pair.stress is not None:
        kpis += _kpi("When the VIX is above 30", _fmt(pair.stress.value),
                     f"n {pair.stress.n_observations:,} days")

    findings = "".join(
        f'<div class="finding {_esc(tone)}"><b>{_esc(title)}</b>{_esc(text)}</div>'
        for title, text, tone in summary_lines(pair)
    )
    lines = charts.rolling_lines(
        pd.DataFrame({"1 year": pair.rolling, "Long run": pair.long_run}),
        f"{pair.name}: rolling one-year and long-run correlation",
    )
    forecast_rows = "".join(
        f"<tr><td>{_esc(f.horizon_label)}</td><td class='num'><b>{_fmt(f.forecast)}</b></td>"
        f"<td class='num'>{_fmt(f.low)} to {_fmt(f.high)}</td>"
        f"<td class='num'>{f.weight:.2f}</td><td class='num'>{_fmt(f.recent)}</td>"
        f"<td class='num'>{_fmt(f.long_run)}</td>"
        f"<td class='num'>{_fmt(f.median_abs_error, '.2f')}</td>"
        f"<td class='num'>{_fmt(f.recent_median_abs_error, '.2f')}</td>"
        f"<td class='num'>{f.n_evaluated}</td></tr>"
        for f in pair.forecasts
    )
    stress_rows = _figure_rows(pair.regimes)
    extra = []
    if pair.asymmetry is not None:
        x = pair.asymmetry
        extra.append(f"Days both fell {_fmt(x.down)} vs days both rose {_fmt(x.up)} "
                     f"(adjusted p {_fmt(x.p_adjusted, '.3f')}).")
    for t in pair.tails:
        extra.append(f"Worst {t.quantile:.0%} of days together: {t.lower:.2f} of the time vs "
                     f"{t.lower_benchmark:.2f} under a normal model ({t.joint_lower_days} days).")
    years = "".join(
        f"<tr><td>{_esc(f.label)}</td><td class='num'>{_fmt(f.value)}</td>"
        f"<td class='num'>{f.n_observations}</td><td class='muted'>{_esc(f.detail)}</td></tr>"
        for f in reversed(pair.calendar_years)
    )
    notes = "".join(f"<li>{_esc(n)}</li>" for n in pair.notes)
    limits = "".join(f"<li>{_esc(t)}</li>" for t in LIMITATIONS)
    stamp = pair.generated_at.strftime("%Y-%m-%d %H:%M UTC") if pair.generated_at else ""
    view_name = "local currencies" if pair.view == "local" else f"{pair.view} (base currency)"
    title = f"Correlation: {pair.name}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} &middot; Nikolay Gelshtein</title>
<style>{PAGE_CSS}{charts.chart_css()}</style>
<script type="application/json" id="run-meta">{_esc(json.dumps(meta))}</script>
</head>
<body><div class="wrap">
<header>
  <div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
  <h1>{_esc(title)}</h1>
  <p class="muted">Returns in {_esc(view_name)}, {_esc(pair.start.date())} to
  {_esc(pair.end.date())} ({pair.n_days:,} common trading days, limited by
  {_esc(pair.limited_by)}). Generated {_esc(stamp)}.</p>
</header>
<div class="kpis">{kpis}</div>
<section class="card"><h2>Reading</h2><div class="findings">{findings}</div></section>
<section class="card">
  <h2>How it moved</h2>
  <p class="muted">The correlation over the trailing year, against the long-run figure using
  all history up to each date. The gap between the two is what the forecast closes part of.</p>
  {lines}
</section>
<section class="card">
  <h2>Forecast</h2>
  <p class="muted">The last {ANCHOR_WINDOW} days pulled toward the long run, with the weight
  fitted on this pair's earlier outcomes at each horizon, using only outcomes known at the time.
  The range shifts today's forecast by the 10th and 90th percentile of past forecast errors.
  Twelve-month outcomes overlap, so they rest on fewer independent years than the count
  suggests.</p>
  <div class="scroll"><table><tr><th>Horizon</th><th class="num">Forecast</th>
  <th class="num">80% range</th><th class="num">Weight on last year</th>
  <th class="num">Last year</th><th class="num">Long run</th>
  <th class="num">Typical miss</th><th class="num">Repeat last year</th>
  <th class="num">Forecasts</th></tr>{forecast_rows}</table></div>
</section>
<section class="card">
  <h2>The number, measured several ways</h2>
  <div class="scroll"><table><tr><th>Measure</th><th class="num">Correlation</th>
  <th class="num">Days</th><th></th></tr>{_figure_rows(pair.figures + pair.other_view)}</table></div>
</section>
<section class="card">
  <h2>In stress</h2>
  <div class="scroll"><table><tr><th>Regime</th><th class="num">Correlation</th>
  <th class="num">Days</th><th></th></tr>{stress_rows}</table></div>
  <ul class="muted">{''.join(f'<li>{_esc(e)}</li>' for e in extra)}</ul>
</section>
<section class="card">
  <h2>By calendar year</h2>
  <details open><summary>Newest first</summary><table><tr><th>Year</th>
  <th class="num">Correlation</th><th class="num">Days</th><th></th></tr>{years}</table></details>
</section>
<section class="card">
  <h2>Data and method</h2>
  <p class="muted">{_esc(pair.convention)}</p>
  {f'<ul class="muted">{notes}</ul>' if notes else ''}
  <h3>Limitations</h3><ul class="muted">{limits}<li>A range from past errors assumes the future
  misses like the past; a regime the history has not seen can fall outside it.</li></ul>
</section>
<footer>Research output. Not investment advice. No orders are placed.<br>
Figures are point-in-time reads from public data and may be stale or wrong.</footer>
</div></body></html>"""


def list_runs(directory: Path | str = REPORT_DIR) -> dict[str, list[ArchivedRun]]:
    archive = Path(directory) / ARCHIVE_DIR
    runs: dict[str, list[ArchivedRun]] = {}
    for path in sorted(archive.glob("*.html")) if archive.exists() else []:
        match = _META.search(path.read_text(encoding="utf-8", errors="replace"))
        if not match:
            continue
        try:
            meta = json.loads(html.unescape(match.group(1)))
        except json.JSONDecodeError:
            continue
        runs.setdefault(str(meta.get("universe", "?")), []).append(ArchivedRun(path, meta))
    for items in runs.values():
        items.sort(key=lambda r: r.meta.get("generated", ""), reverse=True)
    return dict(sorted(runs.items()))


def write_index(directory: Path | str = REPORT_DIR) -> Path:
    root = Path(directory)
    sections = []
    for universe, items in list_runs(root).items():
        rows = []
        for run in items:
            m = run.meta
            href = _esc(run.path.relative_to(root).as_posix())
            rows.append(
                f"<tr><td><a href=\"{href}\">{_esc(m.get('generated', '')[:16].replace('T', ' '))}"
                f"</a></td><td>{_esc(m.get('base', ''))}</td>"
                f"<td>{_esc(m.get('start', ''))} to {_esc(m.get('end', ''))}</td>"
                f"<td class='num'>{_esc(m.get('n', ''))}</td>"
                f"<td class='num'>{_fmt(m.get('average_correlation'))}</td>"
                f"<td class='num'>{_fmt(m.get('average_correlation_vix_high'))}</td>"
                f"<td class='num'>{_esc(m.get('signal', ''))}</td>"
                f"<td>{_esc(m.get('oos_text', m.get('oos_best', '')))}</td></tr>"
            )
        sections.append(
            f'<section class="card"><h2>{_esc(universe)}</h2><div class="scroll"><table><tr>'
            f"<th>Generated</th><th>Base</th><th>Sample</th><th class='num'>Days</th>"
            f"<th class='num'>Avg corr</th><th class='num'>Avg corr, VIX &gt; 30</th>"
            f"<th class='num'>Signals</th><th>Out of sample</th></tr>{''.join(rows)}"
            f"</table></div></section>"
        )
    body = "".join(sections) or (
        '<section class="card"><p class="muted">No correlation reports yet. Run '
        "<code>python correlation.py --universe cross_asset --html</code>.</p></section>"
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Correlation reports &middot; Nikolay Gelshtein</title>
<style>{PAGE_CSS}</style></head>
<body><div class="wrap"><header><div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
<h1>Correlation reports</h1><p class="muted">Every saved correlation run, newest first. Separate
from the single-ticker research reports in reports/.</p></header>
{body}
<footer>Research output. Not investment advice. No orders are placed.</footer>
</div></body></html>"""
    path = root / INDEX_FILE
    path.write_text(page, encoding="utf-8")
    return path
