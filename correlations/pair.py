"""Everything about the correlation of two assets, and an educated guess at what comes next.

    python correlation.py --pair AAPL MSFT --html

The pair is run through the same pipeline as a universe (cache, FX, calendar
alignment, regimes, tails), on the pair's own common history, which is usually
far longer than a universe's: AAPL-MSFT goes back to 1986. On top of that:

- **Calendar-year correlations**, the plainest picture of how much it moves.
- **A forecast for the next month and the next twelve months** from the
  anchored estimator (correlations/anchoring.py): the last year pulled toward
  the pair's own long-run correlation, with the weight fitted on this pair's
  history at that horizon.
- **A range** from the anchored forecast's own past out-of-sample errors: the
  10th and 90th percentile of (realised - forecast) over every earlier origin,
  added to today's forecast. Twelve-month outcomes overlap (one per month), so
  their errors are correlated and the range rests on fewer independent years
  than the count suggests; the report says so.
- **A stress figure**: the correlation when the VIX was above 30. Diversification
  matters on bad days, and that is when correlations are highest, so risk
  should be sized on this number, not the calm one.

Research output only -- not a recommendation, and nothing here places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from agents.ta_analyst import fetch_price_history
from correlations import data, fred
from correlations.analysis import CorrelationAnalysis, run_analysis
from correlations.anchoring import ANCHOR_WINDOW, MIN_FIT_PERIODS, anchor_path
from correlations.regimes import AsymmetryPair, TailPair
from correlations.report import LIMITATIONS

# Forecast horizons, in months: the allocation's rebalancing step and a year.
PAIR_HORIZONS = (1, 12)

# The forecast range covers this share of past out-of-sample errors.
BAND_QUANTILES = (0.10, 0.90)

# Fewer evaluated forecasts than this and no range is given.
MIN_BAND_PERIODS = 24

# A calendar year with fewer common trading days is shown but flagged.
MIN_YEAR_DAYS = 60

ROLLING_WINDOW = 252


@dataclass(frozen=True)
class PairFigure:
    """One correlation figure with what it rests on."""

    label: str
    value: float
    n_observations: int = 0
    detail: str = ""


@dataclass(frozen=True)
class PairForecast:
    horizon_months: int
    forecast: float
    recent: float  # the last 252 days
    long_run: float  # all history so far
    weight: float  # on the recent year, fitted out of sample
    n_fit: int
    low: float = float("nan")  # forecast + 10th percentile of past errors
    high: float = float("nan")  # forecast + 90th percentile
    n_evaluated: int = 0
    median_abs_error: float = float("nan")
    recent_median_abs_error: float = float("nan")  # plain one-year sample, same origins
    long_run_median_abs_error: float = float("nan")
    overlapping: bool = False  # outcomes overlap, so errors are correlated

    @property
    def horizon_label(self) -> str:
        return "next month" if self.horizon_months == 1 else f"next {self.horizon_months} months"


@dataclass(frozen=True)
class PairAnalysis:
    asset_a: str
    asset_b: str
    base_currency: str
    view: str  # the currency view the figures are in ("local" or the base code)
    convention: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    limited_by: str  # what sets the start of the common history
    figures: tuple[PairFigure, ...]
    other_view: tuple[PairFigure, ...]
    calendar_years: tuple[PairFigure, ...]
    regimes: tuple[PairFigure, ...]
    forecasts: tuple[PairForecast, ...]
    stress: PairFigure | None = None
    asymmetry: AsymmetryPair | None = None
    tails: tuple[TailPair, ...] = ()
    notes: tuple[str, ...] = ()
    generated_at: datetime | None = None
    rolling: pd.Series = field(default_factory=pd.Series, repr=False)
    long_run: pd.Series = field(default_factory=pd.Series, repr=False)
    analysis: CorrelationAnalysis | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.asset_a} vs {self.asset_b}"

    def figure(self, label: str) -> PairFigure | None:
        return next((f for f in self.figures if f.label == label), None)

    def forecast(self, horizon_months: int) -> PairForecast | None:
        return next((f for f in self.forecasts if f.horizon_months == horizon_months), None)


def _pair_value(matrix: pd.DataFrame, a: str, b: str) -> float:
    if a in matrix.index and b in matrix.columns:
        return float(matrix.loc[a, b])
    return float("nan")


def _rolling_path(result, a: str, b: str, window: int):
    rolling = next((r for r in result.rolling if r.window == window), None)
    if rolling is None:
        return None, pd.Series(dtype=float)
    path = next((p for p in rolling.paths if {p.asset_a, p.asset_b} == {a, b}), None)
    for key in (f"{a}|{b}", f"{b}|{a}"):
        if key in rolling.series:
            return path, rolling.series[key].dropna()
    return path, pd.Series(dtype=float)


def _forecast(returns: pd.DataFrame, horizon: int, notes: list[str]) -> PairForecast | None:
    try:
        path = anchor_path(returns.to_numpy(), returns.index, ANCHOR_WINDOW, horizon)
    except ValueError as exc:
        notes.append(f"No {horizon}-month forecast: {exc}")
        return None
    forecast = float(path.forecasts[-1][0, 1])
    errors = path.offdiagonal_errors("forecast")[:, 0]
    result = dict(
        horizon_months=horizon, forecast=forecast, recent=float(path.recent[-1][0, 1]),
        long_run=float(path.long_run[-1][0, 1]), weight=float(path.weights[-1]),
        n_fit=int(path.n_fit[-1]), n_evaluated=int(errors.size), overlapping=horizon > 1,
    )
    if int(path.n_fit[-1]) < MIN_FIT_PERIODS:
        notes.append(
            f"{horizon}-month forecast: only {int(path.n_fit[-1])} earlier outcomes, so it is the "
            f"plain {ANCHOR_WINDOW}-day sample (no anchoring yet)."
        )
    if errors.size >= MIN_BAND_PERIODS:
        lo, hi = np.quantile(errors, BAND_QUANTILES)
        result.update(
            low=float(np.clip(forecast + lo, -1.0, 1.0)),
            high=float(np.clip(forecast + hi, -1.0, 1.0)),
            median_abs_error=float(np.median(np.abs(errors))),
            recent_median_abs_error=float(np.median(np.abs(
                path.offdiagonal_errors("recent")[:, 0]))),
            long_run_median_abs_error=float(np.median(np.abs(
                path.offdiagonal_errors("long_run")[:, 0]))),
        )
    else:
        notes.append(f"{horizon}-month forecast: too few past outcomes ({errors.size}) for a range.")
    return PairForecast(**result)


def analyze_pair(
    asset_a: str,
    asset_b: str,
    base: str = "EUR",
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    fetcher: data.Fetcher = fetch_price_history,
    fx_fetcher: data.Fetcher = fred.fetch_fred_series,
    cache_dir: Path = data.CACHE_DIR,
    config: dict | None = None,
    now: datetime | None = None,
) -> PairAnalysis:
    """Full correlation picture of two assets, in the base currency where there is one."""
    a, b = asset_a.strip().upper(), asset_b.strip().upper()
    if a == b:
        raise ValueError("A pair needs two different symbols.")
    universe = data.Universe(
        name=f"{a} vs {b}", symbols=(a, b), market_proxy=data.DEFAULT_MARKET_PROXY,
        groups=MappingProxyType({a: "pair", b: "pair"}), source="--pair",
        description=f"Pair report: {a} and {b} on their common history.",
    )
    analysis = run_analysis(
        universe, base=base, refresh=refresh, start=start, end=end, fetcher=fetcher,
        fx_fetcher=fx_fetcher, cache_dir=cache_dir, config=config, now=now, evaluate=False,
    )
    view = analysis.views[-1]
    daily, weekly = view.frequency("daily"), view.frequency("weekly")
    notes = list(analysis.notes)
    est = daily.estimates
    pearson = next((p for p in est["pearson"].pairs if {p.asset_a, p.asset_b} == {a, b}), None)
    tau = _pair_value(est["kendall"].matrix, a, b)

    year_path, rolling = _rolling_path(daily, a, b, ROLLING_WINDOW)
    quarter_path, _ = _rolling_path(daily, a, b, 60)
    figures = [
        PairFigure("Full history, daily", pearson.correlation if pearson else float("nan"),
                   est["pearson"].n_observations,
                   f"p {pearson.p_value:.1e}" if pearson else ""),
        PairFigure("Spearman (rank), daily", _pair_value(est["spearman"].matrix, a, b),
                   est["spearman"].n_observations, "robust to outliers"),
        PairFigure("Kendall tau, daily", tau, est["kendall"].n_observations,
                   f"implies a Pearson of {math.sin(math.pi * tau / 2):+.2f} if the joint "
                   f"distribution is elliptical" if math.isfinite(tau) else ""),
    ]
    if weekly is not None:
        figures.append(PairFigure("Full history, weekly", _pair_value(
            weekly.estimates["pearson"].matrix, a, b), weekly.estimates["pearson"].n_observations,
            "free of closing-time effects"))
    if year_path is not None:
        figures.append(PairFigure(
            "Last 12 months", year_path.latest, ROLLING_WINDOW,
            f"1-year windows ranged {year_path.minimum:+.2f} to {year_path.maximum:+.2f}"))
    if quarter_path is not None:
        figures.append(PairFigure("Last 3 months", quarter_path.latest, 60,
                                  f"ranged {quarter_path.minimum:+.2f} to {quarter_path.maximum:+.2f}"))
    ewma = est["ewma"]
    figures.append(PairFigure("EWMA, now", _pair_value(ewma.matrix, a, b),
                              int(ewma.parameters.get("effective_sample_size", 0)),
                              "RiskMetrics, half-life about 11 days"))

    other_view = []
    for v in analysis.views[:-1]:
        d = v.frequency("daily")
        if d is not None:
            other_view.append(PairFigure(
                "Full history, daily (local currencies)" if v.name == "local"
                else f"Full history, daily ({v.name})",
                _pair_value(d.estimates["pearson"].matrix, a, b), d.estimates["pearson"].n_observations,
                v.convention))

    returns = daily.returns[[a, b]]
    years = []
    for year, group in returns.groupby(returns.index.year):
        value = float(group[a].corr(group[b])) if len(group) > 2 else float("nan")
        years.append(PairFigure(str(year), value, len(group),
                                "partial year" if len(group) < MIN_YEAR_DAYS else ""))

    regimes, stress, asymmetry, tails = [], None, None, ()
    if daily.regimes is not None:
        for g in daily.regimes.regimes:
            figure = PairFigure(g.label, _pair_value(g.estimate.matrix, a, b),
                                g.estimate.n_observations, g.definition)
            regimes.append(figure)
            if g.name == "vix_high":
                stress = figure
        asymmetry = next((x for x in daily.regimes.asymmetry
                          if {x.asset_a, x.asset_b} == {a, b}), None)
        tails = tuple(t for t in daily.regimes.tails if {t.asset_a, t.asset_b} == {a, b})

    forecasts = tuple(f for f in (_forecast(returns, h, notes) for h in PAIR_HORIZONS) if f)
    long_run = returns[a].expanding(min_periods=ROLLING_WINDOW).corr(returns[b]).dropna()
    limited_by = view.alignment.limiting_symbol
    fx_limited = [c for c in view.conversions if c.before_fx_history]
    if fx_limited:
        series = ", ".join(sorted({s for c in fx_limited for s in c.fx_series}))
        limited_by = f"the FX history ({series}); in local currencies it starts earlier"
    return PairAnalysis(
        asset_a=a, asset_b=b, base_currency=analysis.base_currency, view=view.name,
        convention=view.convention, start=returns.index.min(), end=returns.index.max(),
        n_days=int(len(returns)), limited_by=limited_by,
        figures=tuple(figures), other_view=tuple(other_view), calendar_years=tuple(years),
        regimes=tuple(regimes), forecasts=forecasts, stress=stress, asymmetry=asymmetry,
        tails=tails, notes=tuple(notes), generated_at=analysis.generated_at,
        rolling=rolling, long_run=long_run, analysis=analysis,
    )


# --------------------------------------------------------------------------
# Reading the numbers
# --------------------------------------------------------------------------


def _fmt(value: float, spec: str = "+.2f") -> str:
    return "n/a" if value is None or not math.isfinite(value) else format(value, spec)


def summary_lines(pair: PairAnalysis) -> list[tuple[str, str, str]]:
    """Plain-language reading: (title, text, tone), generated from the figures."""
    out: list[tuple[str, str, str]] = []
    full, year = pair.figure("Full history, daily"), pair.figure("Last 12 months")
    if full and year and math.isfinite(full.value) and math.isfinite(year.value):
        gap = year.value - full.value
        tone = "neutral" if abs(gap) < 0.15 else "risk" if gap > 0 else "good"
        level = ("well below" if gap <= -0.15 else "well above" if gap >= 0.15 else "close to")
        out.append((
            f"The last year is {level} the long run",
            f"Over {pair.n_days:,} common trading days since {pair.start.date()} the daily "
            f"correlation is {full.value:+.2f}; over the last 12 months it is {year.value:+.2f} "
            f"({year.detail}).",
            tone,
        ))
    twelve = pair.forecast(12)
    if twelve is not None:
        band = (f" Shifting it by the errors that past forecasts of this kind made gives an "
                f"80% range of {twelve.low:+.2f} to {twelve.high:+.2f}."
                if math.isfinite(twelve.low) else "")
        better = ""
        if math.isfinite(twelve.median_abs_error):
            better = (f" Typical miss {twelve.median_abs_error:.2f}, against "
                      f"{twelve.recent_median_abs_error:.2f} for simply repeating the last year "
                      f"and {twelve.long_run_median_abs_error:.2f} for the long run alone.")
        out.append((
            f"Educated guess for the next 12 months: {twelve.forecast:+.2f}",
            f"{twelve.weight:.0%} of the weight on the last year ({twelve.recent:+.2f}) and "
            f"{1 - twelve.weight:.0%} on the long run ({twelve.long_run:+.2f}), the split that "
            f"fitted this pair's earlier 12-month outcomes best ({twelve.n_fit} of them, each "
            f"known before the forecast date).{band}{better}",
            "neutral",
        ))
    if pair.stress is not None and math.isfinite(pair.stress.value):
        peak = max((r for r in pair.regimes if r.label not in (
            "Calm", "Volatile", "VIX > 30", "VIX <= 30") and math.isfinite(r.value)),
            key=lambda r: r.value, default=None)
        out.append((
            f"In a sell-off, expect about {pair.stress.value:+.2f}",
            f"On the {pair.stress.n_observations} days the VIX was above 30 the correlation "
            f"was {pair.stress.value:+.2f}"
            + (f"; the highest in a named crisis was {peak.value:+.2f} ({peak.label})"
               if peak else "")
            + ". Diversification is weakest exactly when it is needed, so size risk on this "
              "figure rather than the calm one.",
            "risk",
        ))
    crash = next((t for t in pair.tails if t.quantile == 0.05), None)
    if crash is not None and math.isfinite(crash.lower):
        out.append((
            "How often they crash together",
            f"On each one's worst 5% of days, the other was also in its worst 5% "
            f"{crash.lower:.0%} of the time ({crash.joint_lower_days} days), against "
            f"{crash.lower_benchmark:.0%} under a normal model with the same correlation.",
            "risk" if crash.lower - crash.lower_benchmark >= 0.10 else "neutral",
        ))
    return out


def format_pair(pair: PairAnalysis) -> str:
    rule, heavy = "-" * 78, "=" * 78
    lines = [heavy, f"  PAIR CORRELATION: {pair.name}   (base currency {pair.base_currency})",
             heavy,
             f"  Common history {pair.start.date()} .. {pair.end.date()}, {pair.n_days:,} trading "
             f"days, limited by {pair.limited_by}.",
             f"  {pair.convention}", ""]
    for title, text, _ in summary_lines(pair):
        lines += [f"  * {title}", f"    {text}", ""]

    lines += [rule, "  THE NUMBER", rule]
    for f in pair.figures + pair.other_view:
        detail = f"  ({f.detail})" if f.detail and not f.detail.startswith(("Local", "Base")) \
            else ""
        lines.append(f"    {f.label:<40s}{_fmt(f.value):>7s}   n {f.n_observations:>6,}{detail}")

    lines += ["", rule, "  FORECAST (anchored: last year pulled toward the long run)", rule]
    for f in pair.forecasts:
        band = (f"80% range {_fmt(f.low)} .. {_fmt(f.high)}" if math.isfinite(f.low)
                else "no range (too little history)")
        lines.append(f"    {f.horizon_label:<16s}{_fmt(f.forecast):>7s}   {band}")
        lines.append(f"      weight {f.weight:.2f} on the last year ({_fmt(f.recent)}), "
                     f"{1 - f.weight:.2f} on the long run ({_fmt(f.long_run)}); "
                     f"fitted on {f.n_fit} outcomes")
        if math.isfinite(f.median_abs_error):
            lines.append(f"      typical miss {f.median_abs_error:.2f} (repeat last year "
                         f"{f.recent_median_abs_error:.2f}, long run only "
                         f"{f.long_run_median_abs_error:.2f}) over {f.n_evaluated} forecasts"
                         + ("; overlapping outcomes, so fewer independent years"
                            if f.overlapping else ""))

    lines += ["", rule, "  IN STRESS", rule]
    for r in pair.regimes:
        flag = "  (below 60 days: noisy)" if 0 < r.n_observations < 60 else ""
        lines.append(f"    {r.label:<40s}{_fmt(r.value):>7s}   n {r.n_observations:>6,}{flag}")
    if pair.asymmetry is not None:
        x = pair.asymmetry
        lines.append(f"    Days both fell {_fmt(x.down)} vs days both rose {_fmt(x.up)} "
                     f"(adjusted p {_fmt(x.p_adjusted, '.3f')})")
    for t in pair.tails:
        lines.append(f"    Worst {t.quantile:.0%} days together: {t.lower:.2f} vs normal model "
                     f"{t.lower_benchmark:.2f} ({t.joint_lower_days} days)")

    lines += ["", rule, "  BY CALENDAR YEAR", rule]
    row = []
    for f in pair.calendar_years:
        row.append(f"{f.label} {_fmt(f.value)}{'*' if f.detail else ''}")
        if len(row) == 6:
            lines.append("    " + "   ".join(row))
            row = []
    if row:
        lines.append("    " + "   ".join(row))
    if any(f.detail for f in pair.calendar_years):
        lines.append(f"    * fewer than {MIN_YEAR_DAYS} trading days")

    if pair.notes:
        lines += ["", rule, "  NOTES", rule] + [f"    - {n}" for n in pair.notes]
    lines += ["", rule, "  LIMITATIONS", rule] + [f"    - {t}" for t in LIMITATIONS]
    lines += ["    - A forecast range from past errors assumes the future misses like the past; "
              "a regime the history has not seen can fall outside it.",
              "", heavy, "  Research output only: not financial advice, no orders placed.", heavy]
    return "\n".join(lines)
