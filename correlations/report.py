"""Terminal and JSON output for a correlation analysis.

The terminal report leads with what changes a decision -- which pairs survive
the multiple-comparison correction, which are unstable, where daily and weekly
disagree -- and keeps full matrices to the lower triangle so an 11-asset
universe still fits a terminal.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from agents.risk_manager import MIN_OBSERVATIONS
from backtests.regression import SIGNIFICANCE_LEVEL
from correlations.analysis import CorrelationAnalysis, CurrencyView, FrequencyResult
from correlations.estimators import (
    COVARIANCE_WINDOW,
    DEFAULT_COVARIANCE_METHOD,
    CorrelationEstimate,
    PairStatistic,
)
from correlations.evaluation import FORECAST_ERROR, LONG_ONLY, EstimatorScore
from correlations.synchronicity import EPPS_FLAG_THRESHOLD

WIDTH = 78

# How many rows each "top pairs" list shows. Enough to see the pattern, short
# enough that the report stays readable for a 30-asset peer group.
TOP_N = 5

# Rank vs linear: a gap below this between Pearson and Spearman is noise.
RANK_GAP_NOTE = 0.10

# Winsorized vs raw Pearson: smaller moves are not worth listing.
WINSOR_CHANGE_NOTE = 0.05

LIMITATIONS = (
    "Correlation is not causation: two assets moving together says nothing about why.",
    "Pearson captures linear dependence only; the rank methods catch monotonic but "
    "nonlinear links, and none of them sees a relationship that changes sign.",
    "Every figure depends on the sample period. Correlations change with regimes and "
    "tend to rise in sell-offs, exactly when diversification is needed.",
    "p-values assume independent, identically distributed returns. Real returns cluster "
    "in volatility and have fat tails, so the stated significance is optimistic.",
    "A low p-value (below 0.05 here) is evidence that the correlation is not zero. It does "
    "not say the correlation is large or that it will persist.",
)


def _fmt(value: float, spec: str = "+.2f") -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else format(
        value, spec
    )


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _matrix_lines(matrix: pd.DataFrame) -> list[str]:
    """Lower triangle, one row per asset, columns abbreviated to fit."""
    names = list(matrix.columns)
    label = max(len(n) for n in names)
    cell = 6
    header = " " * (label + 4) + "".join(n[: cell - 1].rjust(cell) for n in names[:-1])
    lines = [header]
    for i, row in enumerate(names):
        if i == 0:
            continue
        cells = "".join(_fmt(matrix.iloc[i, j], "+.2f").rjust(cell) for j in range(i))
        lines.append(f"    {row.ljust(label)}{cells}")
    return lines


def _significance_lines(estimate: CorrelationEstimate) -> list[str]:
    pairs = [p for p in estimate.pairs if not math.isnan(p.p_value)]
    if not pairs:
        return [f"    No testable pairs ({estimate.method})."]
    raw = sum(p.significant for p in pairs)
    adjusted = sum(p.significant_adjusted for p in pairs)
    lines = [
        f"    {estimate.method}: {len(pairs)} pairs tested; {raw} significant at "
        f"p < {SIGNIFICANCE_LEVEL} raw, {adjusted} after Benjamini-Hochberg."
    ]
    lost = sorted(
        (p for p in pairs if p.significant and not p.significant_adjusted),
        key=lambda p: p.p_value,
    )
    if lost:
        lines.append("    Lost after correction (raw p -> adjusted p):")
        for p in lost[:TOP_N]:
            lines.append(
                f"      {p.asset_a}-{p.asset_b}  r {_fmt(p.correlation)}  "
                f"p {p.p_value:.3f} -> {p.p_adjusted:.3f}"
            )
        if len(lost) > TOP_N:
            lines.append(f"      ... and {len(lost) - TOP_N} more")
    insignificant = [p for p in pairs if not p.significant_adjusted]
    if insignificant:
        names = ", ".join(f"{p.asset_a}-{p.asset_b}" for p in insignificant[:8])
        more = f" (+{len(insignificant) - 8})" if len(insignificant) > 8 else ""
        lines.append(f"    Not distinguishable from zero after correction: {names}{more}")
    return lines


def _pair_gaps(a: CorrelationEstimate, b: CorrelationEstimate) -> list[tuple[str, str, float, float]]:
    assets = list(a.assets)
    out = []
    for i, x in enumerate(assets):
        for y in assets[i + 1 :]:
            va, vb = a.pair(x, y), b.pair(x, y)
            if not (math.isnan(va) or math.isnan(vb)):
                out.append((x, y, va, vb))
    out.sort(key=lambda t: -abs(t[3] - t[2]))
    return out


def _frequency_lines(result: FrequencyResult) -> list[str]:
    est = result.estimates
    p = est["pearson"]
    lines: list[str] = []
    span = f"{p.start.date()} .. {p.end.date()}" if p.start is not None else "no data"
    lines += [
        "",
        f"  {result.frequency.upper()} RETURNS   n = {p.n_observations:,}   ({span})",
    ]
    for w in p.warnings:
        lines.append(f"    ! {w}")
    if p.matrix.isna().all().all():
        return lines

    lines += ["", "  Pearson correlation (lower triangle)"] + _matrix_lines(p.matrix)

    lines += ["", "  Significance (each r carries its p-value; BH controls false discoveries)"]
    for method in ("pearson", "spearman", "kendall"):
        lines += _significance_lines(est[method])

    gaps = [g for g in _pair_gaps(p, est["spearman"]) if abs(g[3] - g[2]) >= RANK_GAP_NOTE]
    lines += ["", "  Rank vs linear (Spearman - Pearson)"]
    if gaps:
        lines.append(
            "    Large gaps point to nonlinearity or to a few extreme days driving Pearson:"
        )
        for a, b, pr, sr in gaps[:TOP_N]:
            lines.append(f"      {a}-{b}  Pearson {_fmt(pr)}  Spearman {_fmt(sr)}")
    else:
        lines.append(f"    No pair differs by {RANK_GAP_NOTE:.2f} or more: the links look linear.")

    e = est["ewma"]
    lam = e.parameters.get("lambda")
    ess = e.parameters.get("effective_sample_size")
    lines += ["", f"  EWMA now vs full sample (lambda {lam}, effective n ~ {ess})"]
    for a, b, full, now in _pair_gaps(p, e)[:TOP_N]:
        lines.append(f"      {a}-{b}  full {_fmt(full)}  now {_fmt(now)}  ({_fmt(now - full)})")

    for rolling in result.rolling:
        unit = {"daily": "day", "weekly": "week"}[result.frequency]
        lines += ["", f"  Rolling {rolling.window}-{unit} correlation: widest ranges"]
        if not rolling.paths:
            lines += [f"    ! {w}" for w in rolling.warnings] or ["    No paths."]
            continue
        widest = sorted(rolling.paths, key=lambda q: -q.range)[:TOP_N]
        for q in widest:
            lines.append(
                f"      {q.asset_a}-{q.asset_b}  min {_fmt(q.minimum)}  max {_fmt(q.maximum)}  "
                f"latest {_fmt(q.latest)}  (full {_fmt(q.full_sample)})"
            )

    lines += _structured_lines(result)
    lines += _regime_lines(result)

    if result.lagged:
        sig = [s for s in result.lagged if s.significant_adjusted]
        lines += ["", f"  Lag-1 cross-correlation, cross-continent pairs "
                      f"({len(result.lagged)} tests, {len(sig)} significant after BH)"]
        for s in sorted(sig, key=lambda s: -abs(s.correlation))[:TOP_N]:
            lines.append(
                f"      {s.asset_a}-{s.asset_b}  {s.note}: r {_fmt(s.correlation)}  "
                f"adj p {s.p_adjusted:.3f}"
            )

    if result.outliers:
        lines += ["", f"  Outliers (|robust z| > 6): {len(result.outliers)} flagged, largest:"]
        for o in result.outliers[:TOP_N]:
            lines.append(
                f"      {o.symbol} {o.date.date()}  log return {o.log_return:+.2%}  "
                f"z {o.robust_z:+.1f}"
            )
    zeros = {s: n for s, n in result.zero_returns.items() if n}
    if zeros:
        lines.append(
            "    Zero returns in source data (not created by alignment): "
            + ", ".join(f"{s} {n}" for s, n in zeros.items())
        )

    if result.winsorized:
        moved = [
            g for g in _pair_gaps(p, result.winsorized["pearson"])
            if abs(g[3] - g[2]) >= WINSOR_CHANGE_NOTE
        ]
        lines += ["", "  Winsorized (0.5% per tail) vs raw Pearson"]
        if moved:
            for a, b, raw, clipped in moved[:TOP_N]:
                lines.append(f"      {a}-{b}  raw {_fmt(raw)}  winsorized {_fmt(clipped)}")
        else:
            lines.append(
                f"    No pair moves by {WINSOR_CHANGE_NOTE:.2f} or more: extreme days do not "
                f"drive these correlations."
            )
    return lines


STRUCTURED_LABELS = {
    "ledoit_wolf": "Ledoit-Wolf, constant-correlation target",
    "ledoit_wolf_identity": "Ledoit-Wolf, identity target",
    "denoised": "RMT denoised",
    "pca_factor": "PCA factor model",
    "group_factor": "Market + group factor model",
}


def mean_abs_offdiagonal_gap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Average |a_ij - b_ij| over distinct pairs: how far one matrix sits from another."""
    x, y = a.to_numpy(), b.to_numpy()
    mask = ~np.eye(len(x), dtype=bool)
    gap = np.abs(x - y)[mask]
    return float(np.nanmean(gap)) if gap.size else float("nan")


def average_correlation(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy()
    mask = ~np.eye(len(values), dtype=bool)
    return float(np.nanmean(values[mask])) if mask.any() else float("nan")


def _structured_lines(result: FrequencyResult) -> list[str]:
    sample = result.estimates["pearson"]
    lines = ["", "  Structured estimators (average |change| per pair vs the sample matrix)"]
    for key, label in STRUCTURED_LABELS.items():
        est = result.estimates.get(key)
        if est is None:
            continue
        if est.matrix.isna().all().all():
            lines.append(f"    {label}: not estimable ({est.warnings[-1] if est.warnings else ''})")
            continue
        p = est.parameters
        detail = {
            "ledoit_wolf": lambda: f"intensity {p.get('intensity')}, "
            f"average correlation {p.get('average_correlation')}",
            "ledoit_wolf_identity": lambda: f"intensity {p.get('intensity')}",
            "denoised": lambda: f"{p.get('n_signal')} signal eigenvalue(s) above "
            f"{p.get('signal_edge')}, sigma^2 {p.get('sigma2')}, T/N {p.get('q')}",
            "pca_factor": lambda: f"{p.get('n_factors')} factors explain {p.get('explained'):.0%}",
            "group_factor": lambda: f"factors {p.get('factors')}; explain "
            f"{p.get('explained'):.0%}",
        }[key]()
        gap = mean_abs_offdiagonal_gap(est.matrix, sample.matrix)
        psd = "" if est.is_psd else "  [NOT PSD]"
        lines.append(f"    {label}: {detail}; change {gap:.3f}{psd}")
    if result.clustering is not None:
        lines.append(f"    Cluster order: {' '.join(result.clustering.ordered_labels)}")
    return lines


def _regime_lines(result: FrequencyResult) -> list[str]:
    analysis = result.regimes
    if analysis is None:
        return []
    lines = ["", "  Regimes and stress (Pearson on the subset of days)"]
    for regime in analysis.regimes:
        est = regime.estimate
        if est.matrix.isna().all().all():
            reason = regime.warnings[-1] if regime.warnings else "not estimable"
            lines.append(f"    {regime.label:<34s} n {est.n_observations:>5}  {reason}")
            continue
        flag = "  (below 60: noisy)" if est.n_observations < MIN_OBSERVATIONS else ""
        lines.append(
            f"    {regime.label:<34s} n {est.n_observations:>5}  average correlation "
            f"{_fmt(average_correlation(est.matrix))}{flag}"
        )
    if analysis.shifts:
        lines.append(f"    Losing diversification in stress ({analysis.shifts[0].stress_definition}):")
        for s in analysis.shifts[: TOP_N * 2]:
            lines.append(
                f"      {s.asset_a}-{s.asset_b}  calm {_fmt(s.calm)}  stressed {_fmt(s.stressed)}"
                f"  ({_fmt(s.change)})"
            )
    else:
        lines.append("    No pair's correlation rises by 0.20 or more when the VIX is above 30.")

    significant = sorted(
        (a for a in analysis.asymmetry if a.significant_adjusted), key=lambda a: -a.difference
    )
    lines.append(
        f"    Down-day vs up-day correlation: {len(significant)} of {len(analysis.asymmetry)} "
        f"pairs differ significantly (block bootstrap, Benjamini-Hochberg)"
    )
    for a in significant[:TOP_N]:
        lines.append(
            f"      {a.asset_a}-{a.asset_b}  down {_fmt(a.down)}  up {_fmt(a.up)}  "
            f"(diff {_fmt(a.difference)}, adj p {a.p_adjusted:.3f})"
        )
    excess = [c.excess for c in analysis.conditional
              if c.condition == "both_down" and not np.isnan(c.excess)]
    if excess:
        lines.append(
            f"    Down-day correlation vs a normal model: median excess {np.median(excess):+.2f} "
            f"(fat tails and volatility clustering)"
        )
    tails = sorted(
        (t for t in analysis.tails if t.quantile == 0.05), key=lambda t: -t.lower_excess
    )
    if tails:
        lines.append("    Joint crash risk, worst 5% days (tail dependence vs normal model):")
        for t in tails[:TOP_N]:
            lines.append(
                f"      {t.asset_a}-{t.asset_b}  {t.lower:.2f} vs {t.lower_benchmark:.2f}  "
                f"({t.joint_lower_days} joint days)"
            )
    return lines


def _alignment_lines(view: CurrencyView) -> list[str]:
    log = view.alignment
    if log.common_start is None:
        return ["  ! The series do not overlap at all."]
    lines = [
        f"  Common calendar: {log.common_dates:,} dates, {log.common_start.date()} .. "
        f"{log.common_end.date()}.",
        f"  History limited by {log.limiting_symbol}, whose data starts "
        f"{log.first_dates[log.limiting_symbol].date()}.",
    ]
    missing = {s: len(d) for s, d in log.missing_dates.items() if d}
    if missing:
        lines.append(
            "  Market closed while others traded (those dates dropped for all): "
            + ", ".join(f"{s} {n}" for s, n in sorted(missing.items(), key=lambda t: -t[1]))
        )
    dropped_fx = {c.symbol: len(c.dropped_dates) for c in view.conversions if c.dropped_dates}
    if dropped_fx:
        lines.append(
            "  Dates inside the FX history without an FX quote (dropped before alignment): "
            + ", ".join(f"{s} {n}" for s, n in dropped_fx.items())
        )
    before = {c.symbol: c.before_fx_history for c in view.conversions if c.before_fx_history}
    if before:
        lines.append(
            "  Dates before the FX history starts (cannot be converted): "
            + ", ".join(f"{s} {n}" for s, n in before.items())
        )
    return lines


def verdict(score: EstimatorScore) -> str:
    """One word for a score against the benchmark, after the BH correction."""
    if score.is_benchmark:
        return "benchmark"
    return "better" if score.better else "worse" if score.worse else "no difference"


def evaluation_value(metric: str, value: float) -> str:
    """RMSE per pair as a number, portfolio volatility as a percentage."""
    return _fmt(value, ".3f") if metric == FORECAST_ERROR else _fmt(value, ".2%")


def _evaluation_lines(analysis: CorrelationAnalysis) -> list[str]:
    result = analysis.evaluation
    if result is None:
        return []
    lines = ["", _rule(), "  OUT-OF-SAMPLE: WHICH ESTIMATOR FORECASTS BEST (walk-forward)", _rule()]
    if not result.evaluated:
        return lines + [f"  Not evaluated: {result.warnings[-1] if result.warnings else ''}"]
    bench = next(c.label for c in result.contestants if c.key == result.benchmark)
    lines += [
        f"  {result.n_periods} monthly forecasts, {result.first_forecast.date()} .. "
        f"{result.last_forecast.date()}, scored through {result.evaluation_end.date()}.",
        f"  Each estimator sees only the previous {result.estimation_window} days (the short "
        f"samples fewer). Benchmark: {bench}.",
        "  Significance: Diebold-Mariano against the benchmark (Newey-West), then "
        "Benjamini-Hochberg.",
        f"  {result.currency_convention}",
    ]
    for ranking in result.rankings:
        portfolio = ranking.metric != FORECAST_ERROR
        extra = "  turnover  holdings" if ranking.metric == LONG_ONLY else (
            "     gross" if portfolio else "")
        lines += ["", f"  {ranking.label}",
                  f"     #  {'Estimator':<42s}{'value':>8s}{'vs bench':>10s}{'wins':>6s}"
                  f"{'adj p':>8s}  {'verdict':<14s}{extra}"]
        for s in ranking.scores:
            wins = "" if s.is_benchmark else _fmt(s.win_rate, ".0%")
            p_adj = "" if s.is_benchmark else _fmt(s.p_adjusted, ".3f")
            tail = ""
            if ranking.metric == LONG_ONLY:
                tail = f"{_fmt(s.turnover, '9.3f')} {_fmt(s.effective_holdings, '8.1f')}"
            elif portfolio:
                tail = f"{_fmt(s.gross_exposure, '9.2f')}"
            lines.append(
                f"    {s.rank:>2}  {s.label:<42s}{evaluation_value(ranking.metric, s.value):>8s}"
                f"{_fmt(s.relative_to_benchmark, '+.1%'):>10s}{wins:>6s}{p_adj:>8s}  "
                f"{verdict(s):<14s}{tail}"
            )
    note = result.concentration_note()
    if note:
        lines += ["", f"  Note: {note}"]
    lines += [f"  Note: {w}" for w in result.warnings]
    lines += [
        "",
        f"  estimate_covariance() defaults to {DEFAULT_COVARIANCE_METHOD!r} on "
        f"{COVARIANCE_WINDOW} days, the choice this evaluation supports.",
    ]
    return lines


def format_analysis(analysis: CorrelationAnalysis) -> str:
    u = analysis.universe
    lines = [
        _rule("="),
        f"  CORRELATION ANALYSIS: {u.name}   (base currency {analysis.base_currency})",
        _rule("="),
        f"  Universe ({u.source}): {', '.join(u.symbols)}",
    ]
    if analysis.window_start or analysis.window_end:
        lines.append(
            f"  Window: {analysis.window_start or 'start'} .. {analysis.window_end or 'latest'}"
        )
    lines.append(f"  Minimum sample for any estimate: {MIN_OBSERVATIONS} observations.")
    for note in analysis.notes:
        lines.append(f"  Note: {note}")

    for view in analysis.views:
        title = "LOCAL CURRENCY" if view.name == "local" else f"BASE CURRENCY {view.name}"
        lines += ["", _rule(), f"  {title}", _rule(), f"  {view.convention}", ""]
        for check in view.fx_timing:
            corr = "  ".join(f"lag {k:+d}: {v:+.3f}" for k, v in check.correlations.items())
            status = "OK" if check.aligned and not check.note else "CHECK"
            lines.append(
                f"  FX timing [{status}] {check.proxy} vs {check.fx_symbol} "
                f"(n {check.n_observations:,}): {corr}"
            )
            if check.note:
                lines.append(f"    ! {check.note}")
        lines += _alignment_lines(view)
        for result in view.frequencies:
            lines += _frequency_lines(result)
        if view.frequency("daily") and view.frequency("weekly"):
            lines += ["", f"  Daily vs weekly (flag when they differ by more than "
                          f"{EPPS_FLAG_THRESHOLD:.2f})"]
            if view.epps:
                by_cause: dict[str, list] = {}
                for f in view.epps:
                    by_cause.setdefault(f.likely_cause, []).append(f)
                for cause, flags in by_cause.items():
                    lines.append(f"    Likely cause: {cause}")
                    for f in flags[:TOP_N * 2]:
                        lines.append(
                            f"      {f.asset_a}-{f.asset_b}  daily {_fmt(f.daily)}  "
                            f"weekly {_fmt(f.weekly)}"
                        )
                    if len(flags) > TOP_N * 2:
                        lines.append(f"      ... and {len(flags) - TOP_N * 2} more")
                lines.append(
                    "    For timing-driven gaps the weekly figure is the better guide: daily "
                    "returns are not measured at the same moment."
                )
            else:
                lines.append("    No pair flagged.")

    if analysis.views and len(analysis.views) > 1:
        local = analysis.views[0].frequency("daily")
        based = analysis.views[1].frequency("daily")
        if local and based:
            gaps = [
                g for g in _pair_gaps(local.estimates["pearson"], based.estimates["pearson"])
                if abs(g[3] - g[2]) >= WINSOR_CHANGE_NOTE
            ]
            lines += ["", _rule(), "  CURRENCY EFFECT (daily Pearson, local vs base)", _rule()]
            if gaps:
                for a, b, lv, bv in gaps[:TOP_N * 2]:
                    lines.append(
                        f"      {a}-{b}  local {_fmt(lv)}  {analysis.base_currency} {_fmt(bv)}"
                    )
            else:
                lines.append("    No pair moves by 0.05 or more when expressed in the base currency.")

    lines += _evaluation_lines(analysis)

    lines += ["", _rule(), "  DATA", _rule()]
    lines += [f"    {entry}" for entry in analysis.fetch_log]

    lines += ["", _rule(), "  LIMITATIONS", _rule()]
    lines += [f"    - {text}" for text in LIMITATIONS]
    lines += [
        "",
        _rule("="),
        "  Research output only: not financial advice, no orders placed.",
        "  nova-research | Nikolay Gelshtein",
        _rule("="),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def to_jsonable(value):
    """Convert analysis objects to JSON-safe structures.

    NaN becomes null (NaN is not valid JSON). Return series and rolling paths
    are dropped: they are thousands of numbers per pair, and the summaries
    carry what a consumer needs. Fields marked repr=False are the large ones.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name)) for f in fields(value) if f.repr
        }
    if isinstance(value, pd.DataFrame):
        return {
            str(r): {str(c): to_jsonable(v) for c, v in row.items()}
            for r, row in value.iterrows()
        }
    if isinstance(value, pd.Series):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else round(number, 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def summary_pairs(estimate: CorrelationEstimate) -> list[PairStatistic]:
    """Pairs sorted strongest first; convenience for callers."""
    return sorted(estimate.pairs, key=lambda p: -abs(p.correlation))
