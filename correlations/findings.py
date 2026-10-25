"""Plain-language findings from a correlation analysis.

Each finding is generated from a threshold on a measured figure and names
that figure, in the same spirit as agents/swot.py: a finding that cannot point
at its number cannot be checked or argued with.

The primary view is the base-currency view when there is one (it is what the
investor actually experiences) and the local view otherwise. Daily figures
lead because they carry the regime and tail analysis; weekly figures are
cited where non-synchronous closes make daily figures unreliable.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from correlations.analysis import CorrelationAnalysis, CurrencyView
from correlations.evaluation import FORECAST_ERROR, LONG_ONLY
from correlations.report import average_correlation

# A pair counts as a diversifier when its full-sample correlation with the
# market proxy is at or below this. 0.3 leaves at least ~90% of an asset's
# variance unexplained by the market.
DIVERSIFIER_MAX_CORRELATION = 0.30

# Excess over the Gaussian benchmark worth reporting for down-day correlation
# and tail co-crash frequency.
EXCESS_NOTE = 0.15

# Local-vs-base gap worth reporting as a currency effect.
CURRENCY_NOTE = 0.10

TOP = 5


@dataclass(frozen=True)
class Finding:
    title: str
    text: str
    tone: str = "neutral"  # "risk", "good", "neutral"


def primary_view(analysis: CorrelationAnalysis) -> CurrencyView:
    return analysis.views[-1]


def _pairs(items, fmt) -> str:
    return "; ".join(fmt(i) for i in items)


def _evaluation_finding(analysis: CorrelationAnalysis) -> Finding | None:
    result = analysis.evaluation
    if result is None or not result.evaluated:
        return None
    errors, long_only = result.ranking(FORECAST_ERROR), result.ranking(LONG_ONLY)
    best, bench = errors.winner, errors.benchmark
    span = f"{result.first_forecast.year}-{result.last_forecast.year}"
    if best.is_benchmark or not best.better:
        title = f"Out of sample, nothing clearly beats the {result.estimation_window}-day sample"
        lead = (f"Over {result.n_periods} monthly forecasts ({span}) the lowest error was "
                f"{best.label} ({best.value:.3f} per pair against {bench.value:.3f} for the "
                f"sample), not significantly better after the multiple-comparison correction.")
    else:
        title = f"Out of sample, {best.label} forecasts correlations best"
        lead = (f"Over {result.n_periods} monthly forecasts ({span}) its error was "
                f"{best.value:.3f} per pair against {bench.value:.3f} for the "
                f"{result.estimation_window}-day sample ({best.relative_to_benchmark:+.1%}, lower "
                f"in {best.win_rate:.0%} of months, adjusted p {best.p_adjusted:.3f}).")
    portfolio = long_only.score(best.key)
    if portfolio is not None and not portfolio.is_benchmark:
        word = "lower" if portfolio.better else "higher" if portfolio.worse else (
            "not significantly different")
        lead += (f" In a long-only minimum-variance portfolio it realised "
                 f"{portfolio.value:.2%} volatility against {long_only.benchmark.value:.2%} "
                 f"({word}).")
    worse = [s.label for s in errors.scores if s.worse]
    if worse:
        lead += f" Significantly worse than the plain sample at forecasting: {', '.join(worse)}."
    note = result.concentration_note()
    if note:
        lead += f" {note}"
    return Finding(title, lead, "good" if best.better else "neutral")


def build_findings(analysis: CorrelationAnalysis) -> tuple[Finding, ...]:
    view = primary_view(analysis)
    daily = view.frequency("daily")
    findings: list[Finding] = []
    if daily is None or daily.estimates["pearson"].matrix.isna().all().all():
        return (Finding("No estimate", "Too little common history to say anything."),)

    sample = daily.estimates["pearson"]
    n_assets = len(sample.assets)
    proxy = analysis.universe.market_proxy
    currency = "local currencies" if view.name == "local" else view.name

    # 1. How much one factor dominates.
    denoised = daily.estimates.get("denoised")
    eig = np.sort(np.linalg.eigvalsh(sample.matrix.to_numpy()))[::-1]
    share = eig[0] / n_assets
    signal = denoised.parameters.get("n_signal") if denoised is not None else None
    findings.append(Finding(
        "One common factor dominates" if share > 0.4 else "Structure is spread out",
        f"The largest eigenvalue carries {share:.0%} of the total variance across "
        f"{n_assets} assets (average pairwise correlation {average_correlation(sample.matrix):+.2f}, "
        f"daily, in {currency}). "
        + (f"Random matrix theory separates {signal} eigenvalue(s) from noise: that is "
           f"roughly how many independent bets this universe offers." if signal else ""),
        "risk" if share > 0.4 else "neutral",
    ))

    # 2. Diversifiers against the market proxy.
    if proxy in sample.assets:
        to_proxy = sample.matrix[proxy].drop(proxy).sort_values()
        diversifiers = to_proxy[to_proxy <= DIVERSIFIER_MAX_CORRELATION]
        regimes = daily.regimes
        stressed = regimes.regime("vix_high") if regimes else None
        if len(diversifiers):
            def describe(name: str) -> str:
                text = f"{name} {to_proxy[name]:+.2f}"
                if stressed is not None and not stressed.estimate.matrix.isna().all().all():
                    text += f" (VIX > 30: {stressed.estimate.pair(proxy, name):+.2f})"
                return text

            findings.append(Finding(
                f"Diversifiers against {proxy}",
                f"Assets with a full-sample correlation to {proxy} of "
                f"{DIVERSIFIER_MAX_CORRELATION:.1f} or less: "
                + ", ".join(describe(n) for n in diversifiers.index[:8]) + ".",
                "good",
            ))
        else:
            findings.append(Finding(
                f"No diversifier against {proxy}",
                f"Every asset correlates more than {DIVERSIFIER_MAX_CORRELATION:.1f} with "
                f"{proxy} over the full sample.",
                "risk",
            ))

    # 3. Diversification lost in stress, and whether it beats the normal model.
    if daily.regimes is not None:
        r = daily.regimes
        calm, high = r.regime("vix_low"), r.regime("vix_high")
        if calm is not None and high is not None and r.shifts:
            findings.append(Finding(
                f"{len(r.shifts)} pair(s) lose diversification when the VIX is above 30",
                f"Average correlation rises from {average_correlation(calm.estimate.matrix):+.2f} "
                f"(n {calm.n_observations}) to {average_correlation(high.estimate.matrix):+.2f} "
                f"(n {high.n_observations}). Largest increases: "
                + _pairs(r.shifts[:TOP], lambda s: f"{s.asset_a}-{s.asset_b} "
                         f"{s.calm:+.2f} to {s.stressed:+.2f}") + ".",
                "risk",
            ))
        asym = [a for a in r.asymmetry if a.significant_adjusted]
        down_first = sorted((a for a in asym if a.difference > 0), key=lambda a: -a.difference)
        up_first = [a for a in asym if a.difference < 0]
        if asym:
            findings.append(Finding(
                "Some pairs move together more on the way down than on the way up",
                f"{len(down_first)} pair(s) correlate significantly more on days both fell than "
                f"on days both rose (block-bootstrap test, Benjamini-Hochberg across "
                f"{len(r.asymmetry)} pairs); {len(up_first)} show the reverse. A symmetric "
                f"distribution, normal or fat-tailed, cannot produce this. Largest: "
                + _pairs(down_first[:TOP], lambda a: f"{a.asset_a}-{a.asset_b} {a.down:+.2f} down "
                         f"vs {a.up:+.2f} up") + ".",
                "risk",
            ))
        else:
            findings.append(Finding(
                "No significant downside asymmetry",
                f"No pair correlates significantly more on joint-down days than on joint-up days "
                f"after the multiple-comparison correction ({len(r.asymmetry)} pairs tested).",
                "neutral",
            ))
        excess = np.array([c.excess for c in r.conditional
                           if c.condition == "both_down" and np.isfinite(c.excess)])
        if excess.size:
            findings.append(Finding(
                "A normal-distribution risk model understates co-movement in sell-offs",
                f"On days both assets fell, the typical pair correlates {np.median(excess):+.2f} "
                f"more than a normal model with the same overall correlation predicts "
                f"({(excess >= EXCESS_NOTE).sum()} of {excess.size} pairs by "
                f"{EXCESS_NOTE:.2f} or more). Mostly fat tails and volatility clustering rather "
                f"than asymmetry, but the consequence is the same: covariance-based risk "
                f"figures are too optimistic on bad days.",
                "risk",
            ))
        tails = sorted(
            (t for t in r.tails if t.quantile == 0.05 and t.lower_excess >= EXCESS_NOTE),
            key=lambda t: -t.lower_excess,
        )
        if tails:
            findings.append(Finding(
                "Joint crashes happen more often than correlation implies",
                f"{len(tails)} pair(s) land in their worst 5% of days together at least "
                f"{EXCESS_NOTE:.2f} more often (as a share) than a normal model with the same "
                f"correlation predicts. A covariance matrix alone understates this risk. "
                f"Largest: " + _pairs(tails[:TOP], lambda t: f"{t.asset_a}-{t.asset_b} "
                f"{t.lower:.2f} vs {t.lower_benchmark:.2f}") + ".",
                "risk",
            ))
        windows = [g for g in r.regimes if g.name not in ("calm", "volatile", "vix_high",
                                                            "vix_low")]
        covered = [g for g in windows if not g.estimate.matrix.isna().all().all()]
        if covered:
            findings.append(Finding(
                "Named stress windows",
                "Average correlation: " + _pairs(covered, lambda g: f"{g.label} "
                f"{average_correlation(g.estimate.matrix):+.2f} (n {g.n_observations}"
                + (", below 60: noisy" if g.n_observations < 60 else "") + ")")
                + f"; full sample {average_correlation(sample.matrix):+.2f}.",
                "neutral",
            ))

    # 4. Significance after the multiple-comparison correction.
    tested = [p for p in sample.pairs if np.isfinite(p.p_value)]
    lost = [p for p in tested if p.significant and not p.significant_adjusted]
    zero = [p for p in tested if not p.significant_adjusted]
    findings.append(Finding(
        "Statistical significance",
        f"{len(tested)} pairs tested; {sum(p.significant_adjusted for p in tested)} remain "
        f"significant after the Benjamini-Hochberg correction. "
        + (f"{len(lost)} were significant only before correction: "
           + _pairs(lost[:TOP], lambda p: f"{p.asset_a}-{p.asset_b} r {p.correlation:+.2f}")
           + ". " if lost else "No pair loses significance through the correction. ")
        + (f"Not distinguishable from zero: "
           + ", ".join(f"{p.asset_a}-{p.asset_b}" for p in zero[:8])
           + (f" (+{len(zero) - 8})" if len(zero) > 8 else "") + "." if zero else ""),
        "neutral",
    ))

    # 5. Instability: the rolling range and where EWMA sits now.
    longest = next((r for r in reversed(daily.rolling) if r.paths), None)
    if longest is not None:
        widest = sorted(longest.paths, key=lambda q: -q.range)[:TOP]
        findings.append(Finding(
            "Correlations are unstable",
            f"Over rolling {longest.window}-day windows the widest swings were: "
            + _pairs(widest, lambda q: f"{q.asset_a}-{q.asset_b} from {q.minimum:+.2f} to "
                     f"{q.maximum:+.2f} (now {q.latest:+.2f})")
            + ". A single full-sample number hides this.",
            "risk",
        ))

    # 6. Which estimator forecasts best out of sample.
    finding = _evaluation_finding(analysis)
    if finding is not None:
        findings.append(finding)

    # 7. Currency effect.
    local = analysis.view("local")
    if local is not None and view.name != "local":
        ld, bd = local.frequency("weekly"), view.frequency("weekly")
        if ld and bd:
            lm, bm = ld.estimates["pearson"].matrix, bd.estimates["pearson"].matrix
            gaps = []
            names = list(bm.columns)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    g = bm.loc[a, b] - lm.loc[a, b]
                    if np.isfinite(g) and abs(g) >= CURRENCY_NOTE:
                        gaps.append((a, b, lm.loc[a, b], bm.loc[a, b]))
            gaps.sort(key=lambda t: -abs(t[3] - t[2]))
            if gaps:
                findings.append(Finding(
                    f"Currency changes the picture for a {view.name} investor",
                    f"{len(gaps)} pair(s) move by {CURRENCY_NOTE:.2f} or more when returns are "
                    f"measured in {view.name} (weekly, which is free of timing effects). "
                    f"Largest: " + _pairs(gaps[:TOP], lambda t: f"{t[0]}-{t[1]} "
                    f"{t[2]:+.2f} local, {t[3]:+.2f} {view.name}") + ".",
                    "risk",
                ))

    # 8. Non-synchronous closes.
    if view.epps:
        findings.append(Finding(
            "Daily figures understate some cross-continent links",
            "Daily and weekly correlations differ by more than 0.15 for: "
            + _pairs(view.epps[:TOP], lambda f: f"{f.asset_a}-{f.asset_b} daily {f.daily:+.2f}, "
                     f"weekly {f.weekly:+.2f}")
            + ". For these pairs the weekly figure is the better guide.",
            "neutral",
        ))
    return tuple(findings)
