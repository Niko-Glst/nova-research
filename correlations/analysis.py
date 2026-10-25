"""Run a full correlation analysis on a universe.

One call fetches every series through the cache, builds the local-currency and
base-currency views, aligns calendars, takes daily and weekly log returns, and
runs every estimator on each combination, plus clustering, and on daily data
the regime, stress-window and tail analysis. On the investor's view (the base
currency when there is one) it then runs the walk-forward evaluation
(correlations/evaluation.py). The result is one frozen object that the
terminal formatter, JSON output and HTML report all read from.

`load_returns` is the same data pipeline without the analysis: aligned,
currency-converted log returns for any list of symbols, which is what the
allocation tool feeds to estimators.estimate_covariance.

Regimes are classified from the market proxy's *local-currency* returns and
the VIX in both views: market stress is a property of the market, not of the
investor's currency. The market factor in the factor model is the proxy's
return in the view's own currency.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from agents.ta_analyst import fetch_price_history
from correlations import data, fred
from correlations.data import AlignmentLog, ConversionLog, Listing, Outlier, Universe
from correlations.clustering import Clustering, cluster
from correlations.evaluation import WalkForwardResult, run_walk_forward
from correlations.stats import SCIPY_FIX, scipy_status
from correlations.estimators import (
    ROLLING_WINDOWS,
    CorrelationEstimate,
    PairStatistic,
    RollingCorrelation,
    denoised,
    ewma,
    group_factor,
    kendall,
    ledoit_wolf_estimate,
    pca_factor,
    pearson,
    rolling_paths,
    spearman,
)
from correlations.regimes import RegimeAnalysis, analyze_regimes
from correlations.synchronicity import (
    EppsFlag,
    cross_continent_pairs,
    epps_flags,
    lagged_correlations,
)

LOCAL = "local"

# Fear gauge for the VIX regime split, fetched like any other price series.
VIX_SYMBOL = "^VIX"


@dataclass(frozen=True)
class FrequencyResult:
    """Every estimate for one currency view at one return frequency."""

    frequency: str
    returns: pd.DataFrame = field(repr=False)
    estimates: Mapping[str, CorrelationEstimate] = field(
        default_factory=lambda: MappingProxyType({})
    )
    rolling: tuple[RollingCorrelation, ...] = ()
    lagged: tuple[PairStatistic, ...] = ()
    outliers: tuple[Outlier, ...] = ()
    zero_returns: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    # Same estimators on winsorized returns, when requested. Rank methods are
    # left out: clipping is monotonic, so ranks barely change.
    winsorized: Mapping[str, CorrelationEstimate] = field(
        default_factory=lambda: MappingProxyType({})
    )
    notes: tuple[str, ...] = ()
    clustering: Clustering | None = None  # on the sample Pearson matrix
    regimes: RegimeAnalysis | None = None  # daily only
    # The market factor's return over the same intervals (the group factor
    # model and the walk-forward need it).
    market: pd.Series | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _Context:
    """What every view needs beyond its own prices."""

    regions: Mapping[str, str]
    groups: Mapping[str, str]
    proxy: str
    proxy_close: pd.Series  # local currency
    vix: pd.Series | None
    windows: Mapping[str, Mapping]
    winsorize_returns: bool


@dataclass(frozen=True)
class CurrencyView:
    """The universe expressed in one currency convention."""

    name: str  # "local" or the base currency code, e.g. "EUR"
    convention: str
    alignment: AlignmentLog
    frequencies: tuple[FrequencyResult, ...]
    epps: tuple[EppsFlag, ...] = ()
    conversions: tuple[ConversionLog, ...] = ()
    fx_timing: tuple[data.FxTimingCheck, ...] = ()

    def frequency(self, name: str) -> FrequencyResult | None:
        return next((f for f in self.frequencies if f.frequency == name), None)


@dataclass(frozen=True)
class CorrelationAnalysis:
    universe: Universe
    base_currency: str
    listings: Mapping[str, Listing]
    views: tuple[CurrencyView, ...]
    fetch_log: tuple[str, ...] = ()
    generated_at: datetime | None = None
    window_start: str | None = None
    window_end: str | None = None
    winsorized: bool = False
    notes: tuple[str, ...] = ()
    # Out-of-sample ranking of the estimators, on the investor's daily view.
    evaluation: WalkForwardResult | None = None

    def view(self, name: str) -> CurrencyView | None:
        return next((v for v in self.views if v.name == name), None)


def _local_convention(listings: Mapping[str, Listing]) -> str:
    by_currency: dict[str, list[str]] = {}
    for listing in listings.values():
        by_currency.setdefault(listing.currency, []).append(listing.symbol)
    parts = "; ".join(f"{c}: {', '.join(s)}" for c, s in sorted(by_currency.items()))
    return (
        f"Local currency: each asset's log returns in its own listing currency ({parts}). "
        f"Returns are taken only on dates every market traded."
    )


def _base_convention(base: str, recipes: Mapping[str, data.FxRecipe]) -> str:
    series = ", ".join(sorted({s for r in recipes.values() for s in r.series})) or "none needed"
    return (
        f"Base currency {base}: prices converted to {base} on each asset's own date using "
        f"Federal Reserve H.10 noon New York rates from FRED ({series}) before alignment, so "
        f"returns include currency moves as a {base} investor experiences them. The noon rate "
        f"is taken four hours before the US close and one and a half after the European "
        f"close, so daily figures carry a small timing gap that weekly figures absorb."
    )


def _interval_returns(close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Log returns of `close` over exactly the intervals between `dates`.

    Matches the asset returns' calendar: a return dated d covers the previous
    date in `dates` to d, so a holiday in the proxy's market does not shift it.
    """
    on_dates = close.reindex(close.index.union(dates)).ffill().reindex(dates)
    returns = np.log(on_dates).diff()
    # Dates before the proxy's history are not filled by ffill; they stay NaN.
    return returns.iloc[1:]


def _analyze_frequency(
    prices: pd.DataFrame,
    frequency: str,
    convention: str,
    ctx: _Context,
) -> FrequencyResult:
    returns = data.log_returns(prices, frequency)
    outliers, outlier_notes = data.find_outliers(returns)
    notes = list(outlier_notes)

    # Market factor: the proxy in this view's currency when it is in the
    # universe, otherwise its local return over the same intervals.
    if ctx.proxy in returns.columns:
        market = returns[ctx.proxy]
    else:
        dates = prices.index if frequency == "daily" else returns.index
        market = _interval_returns(ctx.proxy_close, dates).reindex(returns.index)

    estimates = {
        "pearson": pearson(returns, frequency, convention),
        "spearman": spearman(returns, frequency, convention),
        "kendall": kendall(returns, frequency, convention),
        "ewma": ewma(returns, frequency, currency_convention=convention),
        "ledoit_wolf": ledoit_wolf_estimate(returns, frequency, "constant_correlation",
                                            convention),
        "ledoit_wolf_identity": ledoit_wolf_estimate(returns, frequency, "identity",
                                                     convention),
        "denoised": denoised(returns, frequency, convention),
        "pca_factor": pca_factor(returns, frequency, currency_convention=convention),
        "group_factor": group_factor(returns, ctx.groups, market, frequency, convention,
                                     market_name=ctx.proxy),
    }
    rolling = tuple(
        rolling_paths(returns, window, frequency, convention)
        for window in ROLLING_WINDOWS[frequency]
    )
    pairs = cross_continent_pairs(list(returns.columns), ctx.regions)
    lagged = lagged_correlations(returns, pairs) if pairs else ()

    sample = estimates["pearson"].matrix
    clustering = None
    if not sample.isna().any().any():
        clustering = cluster(sample.to_numpy(), list(sample.columns))
    else:
        notes.append("Clustering skipped: the sample matrix has undefined entries.")

    regime_analysis = None
    if frequency == "daily":
        proxy_local = _interval_returns(ctx.proxy_close, prices.index).reindex(returns.index)
        if proxy_local.notna().sum() >= len(returns) // 2:
            regime_analysis = analyze_regimes(
                returns, proxy_local, ctx.proxy, ctx.vix, ctx.windows, convention
            )
        else:
            notes.append(
                f"Regime analysis skipped: {ctx.proxy} has no history for most of the sample."
            )

    winsorized: dict[str, CorrelationEstimate] = {}
    if ctx.winsorize_returns:
        clipped = data.winsorize(returns)
        winsorized = {
            "pearson": pearson(clipped, frequency, convention),
            "ewma": ewma(clipped, frequency, currency_convention=convention),
        }

    return FrequencyResult(
        frequency=frequency,
        returns=returns,
        estimates=MappingProxyType(estimates),
        rolling=rolling,
        lagged=lagged,
        outliers=outliers,
        zero_returns=MappingProxyType(data.zero_return_counts(returns)),
        winsorized=MappingProxyType(winsorized),
        notes=tuple(notes),
        clustering=clustering,
        regimes=regime_analysis,
        market=market,
    )


def _build_view(
    name: str,
    convention: str,
    closes: Mapping[str, pd.Series],
    frequencies: Sequence[str],
    ctx: _Context,
    conversions: tuple[ConversionLog, ...] = (),
) -> CurrencyView:
    prices, alignment = data.align_prices(closes)
    results = tuple(_analyze_frequency(prices, f, convention, ctx) for f in frequencies)
    by_name = {r.frequency: r for r in results}
    epps: tuple[EppsFlag, ...] = ()
    if "daily" in by_name and "weekly" in by_name:
        epps = epps_flags(
            by_name["daily"].estimates["pearson"],
            by_name["weekly"].estimates["pearson"],
            ctx.regions,
            converted=name != LOCAL,
        )
    return CurrencyView(name, convention, alignment, results, epps, conversions)


Loader = Callable[..., pd.Series]


def _loader(
    refresh: bool,
    cache_dir: Path,
    fetcher: data.Fetcher,
    now: datetime | None,
    start: str | None,
    end: str | None,
    fetch_log: list[str],
) -> Loader:
    """Load one symbol's closes through the cache, logging where they came from."""

    def load(symbol: str, source_fetcher: data.Fetcher = fetcher) -> pd.Series:
        series = data.load_prices(symbol, refresh, cache_dir, source_fetcher, now)
        source = "cache" if series.from_cache else "fetched"
        fetch_log.append(
            f"{symbol}: {source}, {len(series.close)} closes "
            f"{series.close.index.min().date()} .. {series.close.index.max().date()}"
        )
        fetch_log.extend(f"{symbol}: {w}" for w in series.warnings)
        return series.close.loc[start:end]

    return load


def _fx_per_eur(
    recipes: Mapping[str, data.FxRecipe],
    load: Loader,
    fx_fetcher: data.Fetcher,
    fetch_log: list[str],
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Each needed currency per 1 EUR from FRED, with bad quotes removed and logged."""
    components = {
        s: load(f"{fred.PREFIX}{s}", fx_fetcher)
        for s in sorted({s for r in recipes.values() for s in r.series})
    }
    fx: dict[str, pd.Series] = {}
    labels: dict[str, str] = {}
    for major, recipe in recipes.items():
        cleaned, bad = data.remove_reversal_spikes(data.build_fx_per_eur(recipe, components))
        fx[major] = cleaned
        labels[major] = recipe.label
        if bad:
            dates = ", ".join(d.date().isoformat() for d in bad)
            fetch_log.append(
                f"{major} per EUR ({recipe.label}): {len(bad)} bad quote(s) removed "
                f"(one-day spike that fully reverses): {dates}"
            )
    return fx, labels


def run_analysis(
    universe: Universe | str,
    base: str = "EUR",
    frequencies: Sequence[str] = data.FREQUENCIES,
    winsorize_returns: bool = False,
    refresh: bool = False,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    fetcher: data.Fetcher = fetch_price_history,
    fx_fetcher: data.Fetcher = fred.fetch_fred_series,
    cache_dir: Path = data.CACHE_DIR,
    config: dict | None = None,
    now: datetime | None = None,
    evaluate: bool = True,
) -> CorrelationAnalysis:
    """Fetch, convert, align and estimate for every currency view and frequency.

    `evaluate=False` skips the walk-forward evaluation (about ten seconds for
    30 assets and 20 years of daily data).
    """
    config = config if config is not None else data.load_correlation_config()
    if isinstance(universe, str):
        universe = data.resolve_universe(universe, symbols, config)
    base = base.strip().upper()
    for f in frequencies:
        if f not in data.FREQUENCIES:
            raise ValueError(f"Unknown frequency {f!r}; use {data.FREQUENCIES}.")

    listings = {s: data.listing_of(s, config) for s in universe.symbols}
    regions = {s: listing.region for s, listing in listings.items()}

    fetch_log: list[str] = []
    load = _loader(refresh, cache_dir, fetcher, now, start, end, fetch_log)

    closes = {s: load(s) for s in universe.symbols}
    notes: list[str] = []
    available, reason = scipy_status()
    if not available:
        notes.append(
            f"scipy could not be loaded ({reason}); the slower built-in Kendall ran instead. "
            f"To fix, run in nova-research: {SCIPY_FIX}"
        )

    proxy = universe.market_proxy
    proxy_close = closes[proxy] if proxy in closes else load(proxy)
    vix: pd.Series | None = None
    try:
        vix = load(VIX_SYMBOL)
    except Exception as exc:  # the VIX split is optional; the rest of the run is not
        notes.append(f"VIX could not be loaded ({type(exc).__name__}); VIX regimes skipped.")
    ctx = _Context(
        regions=MappingProxyType(regions),
        groups=universe.groups,
        proxy=proxy,
        proxy_close=proxy_close,
        vix=vix,
        windows=MappingProxyType(dict(config.get("stress_windows") or {})),
        winsorize_returns=winsorize_returns,
    )

    views = [_build_view(LOCAL, _local_convention(listings), closes, frequencies, ctx)]

    currencies = [listing.currency for listing in listings.values()]
    recipes = data.required_fx(currencies, base, config)
    majors = {
        str((config.get("minor_units") or {}).get(c, {}).get("major", c)) for c in currencies
    }
    if majors == {base}:
        notes.append(
            f"Every asset is already quoted in {base}, so the base-currency view would be "
            f"identical to the local one and is not repeated."
        )
    else:
        fx, labels = _fx_per_eur(recipes, load, fx_fetcher, fetch_log)
        timing: list[data.FxTimingCheck] = []
        for major, rate in fx.items():
            check = data.fx_timing_check(proxy, proxy_close, f"{major} per EUR", rate)
            timing.append(check)
            if check.note and not check.aligned:
                notes.append(check.note)
        converted: dict[str, pd.Series] = {}
        logs: list[ConversionLog] = []
        for symbol, close in closes.items():
            series, log = data.convert_to_base(
                close, symbol, listings[symbol].currency, base, fx, labels, config
            )
            converted[symbol] = series
            logs.append(log)
        view = _build_view(
            base, _base_convention(base, recipes), converted, frequencies, ctx, tuple(logs)
        )
        views.append(replace(view, fx_timing=tuple(timing)))

        local_end = views[0].alignment.common_end
        base_end = view.alignment.common_end
        if local_end is not None and base_end is not None and base_end < local_end:
            notes.append(
                f"The {base} view ends {base_end.date()}, before the local view "
                f"({local_end.date()}): FRED publishes the H.10 rates weekly, a few days in "
                f"arrears."
            )

    evaluation = None
    investor_daily = views[-1].frequency("daily")
    if evaluate and investor_daily is not None:
        evaluation = run_walk_forward(
            investor_daily.returns,
            investor_daily.market,
            universe.groups,
            currency_convention=views[-1].convention,
        )

    return CorrelationAnalysis(
        universe=universe,
        base_currency=base,
        listings=MappingProxyType(listings),
        views=tuple(views),
        fetch_log=tuple(fetch_log),
        generated_at=now or datetime.now(timezone.utc),
        window_start=start,
        window_end=end,
        winsorized=winsorize_returns,
        notes=tuple(notes),
        evaluation=evaluation,
    )


# --------------------------------------------------------------------------
# Returns for any set of symbols (the allocation tool's input)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnsData:
    """Aligned log returns for a list of symbols, ready for estimate_covariance."""

    returns: pd.DataFrame
    frequency: str
    convention: str  # currency convention, to state alongside any figure
    alignment: AlignmentLog
    conversions: tuple[ConversionLog, ...] = ()
    log: tuple[str, ...] = ()


def load_returns(
    symbols: Sequence[str],
    base: str | None = "EUR",
    frequency: str = "daily",
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    fetcher: data.Fetcher = fetch_price_history,
    fx_fetcher: data.Fetcher = fred.fetch_fred_series,
    cache_dir: Path = data.CACHE_DIR,
    config: dict | None = None,
    now: datetime | None = None,
) -> ReturnsData:
    """Log returns for `symbols` in `base` (None for local currencies).

    The same pipeline as run_analysis: prices through the cache, converted at
    FRED noon rates with bad quotes removed, inner-joined on the dates every
    market traded, then log returns. Nothing is estimated here; pass
    `.returns` to estimators.estimate_covariance.
    """
    config = config if config is not None else data.load_correlation_config()
    unique = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if len(unique) < 2:
        raise ValueError(f"Need at least two symbols, got {list(unique)}.")
    if frequency not in data.FREQUENCIES:
        raise ValueError(f"Unknown frequency {frequency!r}; use {data.FREQUENCIES}.")
    listings = {s: data.listing_of(s, config) for s in unique}

    fetch_log: list[str] = []
    load = _loader(refresh, cache_dir, fetcher, now, start, end, fetch_log)
    closes = {s: load(s) for s in unique}

    conversions: tuple[ConversionLog, ...] = ()
    if base is None:
        convention = _local_convention(listings)
    else:
        base = base.strip().upper()
        recipes = data.required_fx([lst.currency for lst in listings.values()], base, config)
        fx, labels = _fx_per_eur(recipes, load, fx_fetcher, fetch_log) if recipes else ({}, {})
        logs = []
        for symbol, close in closes.items():
            closes[symbol], conversion = data.convert_to_base(
                close, symbol, listings[symbol].currency, base, fx, labels, config
            )
            logs.append(conversion)
        conversions = tuple(logs)
        convention = _base_convention(base, recipes)

    prices, alignment = data.align_prices(closes)
    return ReturnsData(
        returns=data.log_returns(prices, frequency),
        frequency=frequency,
        convention=convention,
        alignment=alignment,
        conversions=conversions,
        log=tuple(fetch_log),
    )
