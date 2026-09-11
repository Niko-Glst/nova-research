"""Recency weighting and coverage-volume analysis for news.

Two ideas from how desks actually read a news feed, neither of which a plain
average of headline scores captures:

**Recency.** A story from three weeks ago is not evidence about today. Articles
are weighted by an exponential decay with a configurable half-life, so a
week-old piece counts roughly half as much as this morning's at the default.

**Coverage volume.** How *much* a name is being written about is often a
stronger signal than whether the coverage reads positive. A sudden tripling of
article count is a real event -- an earnings surprise, a lawsuit, a bid --
whether or not the lexicon can tell you its direction. This module compares a
recent window against an earlier baseline and flags the spike separately from
the sentiment score, because the two answer different questions.

A spike is deliberately NOT folded into the sentiment number. A surge in
coverage is a change in attention; treating it as bullish or bearish on its own
would be a guess. It is reported alongside, for the reader (or a later agent) to
interpret.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents.news_sources import NewsItem

# Days after which an article carries half its original weight.
DEFAULT_HALF_LIFE_DAYS = 7.0

# An article with no timestamp cannot be aged. Rather than dropping it or
# treating it as fresh, it counts at a discount: it is probably recent (feeds
# are ordered newest-first) but that is an assumption, not a fact.
UNDATED_WEIGHT = 0.5

# Windows for the volume comparison, in days.
DEFAULT_RECENT_WINDOW = 3
DEFAULT_BASELINE_WINDOW = 30

# Ratio of recent-to-baseline daily rate that counts as a spike or a lull.
SPIKE_THRESHOLD = 2.0
LULL_THRESHOLD = 0.5

# Below this, a raised ratio is not distinguishable from the rounding noise of
# a short window: evenly spaced daily coverage already lands near 1.4x because
# the recent window spans 3 days and the baseline 27.
ELEVATED_THRESHOLD = 1.5

# A spike computed from very few articles is noise, not news. Two separate
# floors are needed: enough dated articles overall to establish any rate at all,
# and enough articles *in the recent window* to call a burst. Without the second
# floor a single article against a quiet baseline reads as a 3x spike.
MIN_ARTICLES_FOR_SPIKE = 3
MIN_RECENT_ARTICLES_FOR_SPIKE = 3


@dataclass(frozen=True)
class VolumeRead:
    """How current coverage compares to the recent baseline."""

    recent_count: int
    recent_window_days: int
    baseline_count: int
    baseline_window_days: int
    recent_per_day: float
    baseline_per_day: float
    ratio: float  # recent rate / baseline rate; 1.0 means unchanged
    status: str  # "spike" / "elevated" / "normal" / "quiet" / "unknown"
    notes: str


@dataclass(frozen=True)
class NewsSignal:
    """The combined news read: weighted sentiment plus coverage volume."""

    score: float  # -1.0 .. 1.0, recency-weighted
    unweighted_score: float  # plain average, for comparison
    article_count: int
    dated_count: int
    effective_sample: float  # sum of weights: how much evidence this really is
    volume: VolumeRead
    top_positive: list[str] = field(default_factory=list)
    top_negative: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def recency_weight(
    age_days: float | None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS
) -> float:
    """Weight an article by age, halving every `half_life_days`.

    Undated articles get a fixed discount rather than being dropped.
    """
    if age_days is None:
        return UNDATED_WEIGHT
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def analyze_volume(
    items: list[NewsItem],
    recent_window_days: int = DEFAULT_RECENT_WINDOW,
    baseline_window_days: int = DEFAULT_BASELINE_WINDOW,
    now: datetime | None = None,
) -> VolumeRead:
    """Compare recent coverage rate against an earlier baseline.

    The baseline deliberately excludes the recent window, so a genuine spike is
    measured against normal conditions rather than against itself.
    """
    now = now or datetime.now(timezone.utc)

    # A truncated feed cannot support this analysis: it returns only the newest
    # articles, so there is never a baseline behind them and every symbol would
    # read as a spike. Refuse rather than report a guaranteed false positive.
    if items and not any(item.supports_volume_analysis for item in items):
        provider = items[0].provider
        return VolumeRead(
            recent_count=0,
            recent_window_days=recent_window_days,
            baseline_count=0,
            baseline_window_days=baseline_window_days,
            recent_per_day=0.0,
            baseline_per_day=0.0,
            ratio=1.0,
            status="unknown",
            notes=(
                f"Coverage volume cannot be measured from {provider}: it returns "
                f"only the newest articles, with no historical window to compare "
                f"against. Set FINNHUB_API_KEY for volume tracking."
            ),
        )

    dated = [item for item in items if item.published is not None]

    # Even a windowed provider truncates: Finnhub caps a company-news response
    # at a few hundred articles, which on a heavily covered name is filled by
    # the last day or two. The request asked for `baseline_window_days` of
    # history, so if the oldest article is far newer than that, the response was
    # capped and the "missing" baseline is an artifact of the cap, not a
    # genuine absence of earlier coverage.
    oldest_age = max(
        (item.age_days for item in dated if item.age_days is not None), default=0.0
    )

    if dated:
        # The baseline needs *some* span beyond the recent window to be a
        # baseline at all. One extra day is not a comparison.
        if oldest_age < recent_window_days + 1:
            return VolumeRead(
                recent_count=len(dated),
                recent_window_days=recent_window_days,
                baseline_count=0,
                baseline_window_days=baseline_window_days,
                recent_per_day=len(dated) / max(1.0, oldest_age),
                baseline_per_day=0.0,
                ratio=1.0,
                status="unknown",
                notes=(
                    f"{len(dated)} articles span only the last {oldest_age:.1f} days: "
                    f"the provider capped the response before reaching the baseline "
                    f"window, so there is nothing to compare against. Coverage is "
                    f"heavy enough to hit that cap, which is itself a sign of an "
                    f"actively covered name."
                ),
            )

    if len(dated) < MIN_ARTICLES_FOR_SPIKE:
        return VolumeRead(
            recent_count=0,
            recent_window_days=recent_window_days,
            baseline_count=0,
            baseline_window_days=baseline_window_days,
            recent_per_day=0.0,
            baseline_per_day=0.0,
            ratio=1.0,
            status="unknown",
            notes=(
                f"Only {len(dated)} dated articles: too few to judge coverage volume "
                f"(need at least {MIN_ARTICLES_FOR_SPIKE})."
            ),
        )

    recent_count = 0
    baseline_count = 0

    for item in dated:
        age = (now - item.published).total_seconds() / 86400.0
        if age < 0:
            age = 0.0
        if age <= recent_window_days:
            recent_count += 1
        elif age <= baseline_window_days:
            baseline_count += 1

    # Normalize the baseline over the span actually covered by the data, not
    # the span requested. A capped response that reaches back 9 days must be
    # divided by ~6 baseline days, not 27, or its rate is understated and every
    # busy name reads as a spike.
    covered_span = min(float(baseline_window_days), oldest_age)
    baseline_span = max(1.0, covered_span - recent_window_days)
    recent_per_day = recent_count / max(1, recent_window_days)
    baseline_per_day = baseline_count / baseline_span

    if baseline_per_day == 0:
        # No prior coverage to compare against. That is only meaningful if there
        # is a real burst now; otherwise the sample is simply too thin.
        if recent_count >= MIN_RECENT_ARTICLES_FOR_SPIKE:
            ratio = float("inf")
            status = "spike"
            notes = (
                f"{recent_count} articles in the last {recent_window_days} days "
                f"with no coverage in the preceding baseline: a name that was "
                f"not being written about suddenly is."
            )
        else:
            ratio = 1.0
            status = "unknown"
            notes = "Not enough baseline coverage to compare against."
        return VolumeRead(
            recent_count=recent_count,
            recent_window_days=recent_window_days,
            baseline_count=baseline_count,
            baseline_window_days=baseline_window_days,
            recent_per_day=recent_per_day,
            baseline_per_day=baseline_per_day,
            ratio=ratio,
            status=status,
            notes=notes,
        )

    ratio = recent_per_day / baseline_per_day

    # A high ratio off a tiny recent count is arithmetic, not news: one article
    # against a baseline of 0.1/day is "10x" while telling you nothing.
    if ratio >= SPIKE_THRESHOLD and recent_count < MIN_RECENT_ARTICLES_FOR_SPIKE:
        return VolumeRead(
            recent_count=recent_count,
            recent_window_days=recent_window_days,
            baseline_count=baseline_count,
            baseline_window_days=baseline_window_days,
            recent_per_day=recent_per_day,
            baseline_per_day=baseline_per_day,
            ratio=ratio,
            status="normal",
            notes=(
                f"Coverage rate is {ratio:.1f}x baseline, but on only "
                f"{recent_count} recent article(s) -- too thin to call a spike."
            ),
        )

    if ratio >= SPIKE_THRESHOLD:
        status = "spike"
        notes = (
            f"Coverage is running {ratio:.1f}x its baseline rate "
            f"({recent_per_day:.1f} vs {baseline_per_day:.1f} articles/day). "
            f"Something is happening; the sentiment score says what kind, "
            f"this says how loudly."
        )
    elif ratio >= ELEVATED_THRESHOLD:
        status = "elevated"
        notes = (
            f"Coverage is modestly elevated at {ratio:.1f}x baseline "
            f"({recent_per_day:.1f} vs {baseline_per_day:.1f} articles/day)."
        )
    elif ratio <= LULL_THRESHOLD:
        status = "quiet"
        notes = (
            f"Coverage has fallen to {ratio:.1f}x baseline "
            f"({recent_per_day:.1f} vs {baseline_per_day:.1f} articles/day)."
        )
    else:
        status = "normal"
        notes = (
            f"Coverage is running near its baseline rate "
            f"({recent_per_day:.1f} vs {baseline_per_day:.1f} articles/day)."
        )

    return VolumeRead(
        recent_count=recent_count,
        recent_window_days=recent_window_days,
        baseline_count=baseline_count,
        baseline_window_days=baseline_window_days,
        recent_per_day=recent_per_day,
        baseline_per_day=baseline_per_day,
        ratio=ratio,
        status=status,
        notes=notes,
    )


def build_news_signal(
    items: list[NewsItem],
    score_text,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    recent_window_days: int = DEFAULT_RECENT_WINDOW,
    baseline_window_days: int = DEFAULT_BASELINE_WINDOW,
    now: datetime | None = None,
) -> NewsSignal:
    """Score a set of articles with recency weighting, plus a volume read.

    `score_text` is injected rather than imported so this module stays
    independent of which lexicon or model does the scoring.
    """
    volume = analyze_volume(
        items,
        recent_window_days=recent_window_days,
        baseline_window_days=baseline_window_days,
        now=now,
    )

    if not items:
        return NewsSignal(
            score=0.0,
            unweighted_score=0.0,
            article_count=0,
            dated_count=0,
            effective_sample=0.0,
            volume=volume,
            reasons=["No articles available to score."],
        )

    weighted_total = 0.0
    weight_total = 0.0
    plain_total = 0.0
    scored: list[tuple[float, NewsItem]] = []

    for item in items:
        text = f"{item.title} {item.body}".strip()
        score = score_text(text)
        weight = recency_weight(item.age_days, half_life_days)

        weighted_total += score * weight
        weight_total += weight
        plain_total += score
        if score != 0.0:
            scored.append((score, item))

    weighted = weighted_total / weight_total if weight_total else 0.0
    unweighted = plain_total / len(items)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_positive = [item.title for score, item in scored[:3] if score > 0]
    top_negative = [item.title for score, item in reversed(scored[-3:]) if score < 0]

    dated_count = sum(1 for item in items if item.published is not None)

    reasons: list[str] = [
        f"Recency-weighted score {weighted:+.2f} across {len(items)} articles "
        f"(unweighted {unweighted:+.2f}, half-life {half_life_days:.0f}d)."
    ]
    if dated_count < len(items):
        reasons.append(
            f"{len(items) - dated_count} of {len(items)} articles had no "
            f"timestamp and were weighted at {UNDATED_WEIGHT:.0%}."
        )
    if volume.status != "unknown":
        reasons.append(volume.notes)

    return NewsSignal(
        score=max(-1.0, min(1.0, weighted)),
        unweighted_score=max(-1.0, min(1.0, unweighted)),
        article_count=len(items),
        dated_count=dated_count,
        effective_sample=weight_total,
        volume=volume,
        top_positive=top_positive,
        top_negative=top_negative,
        reasons=reasons,
    )
