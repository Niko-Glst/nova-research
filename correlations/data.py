"""Price data for the correlation module: fetch, cache, currency, alignment.

Everything a correlation is computed from passes through here, and most of the
ways a correlation study goes wrong happen at this stage rather than in the
estimator:

- **Levels instead of returns.** Two trending prices correlate strongly by
  construction. Only log returns leave this module.
- **Calendars.** Paris, Zurich and New York close on different holidays, and
  yfinance indexes each series in its own exchange's timezone. Indexes are
  reduced to local exchange dates and inner-joined on the *price* level before
  any return is taken, so every return spans the same calendar interval for
  every asset. A holiday in one market becomes a two-day return for all of
  them. It never becomes a zero return, which is what forward-filling the
  closed market would produce, biasing every correlation toward zero.
- **Currency.** A European investor holding SPY earns its USD return plus the
  move in EURUSD. Prices can be converted to a base currency with official
  Federal Reserve noon rates from FRED (correlations/fred.py explains why not
  Yahoo FX), and the convention travels with every result.

Asset prices are fetched through `agents.ta_analyst.fetch_price_history`
(rate-limited, validated) and FX through `correlations.fred`, one series at a
time, both behind the same local CSV cache under data/cache/.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import yaml

from agents.ta_analyst import fetch_price_history
from common.files import atomic_write, atomic_write_text
from config.settings import load_allowed_sources

ROOT_DIR = Path(__file__).resolve().parent.parent
UNIVERSES_PATH = ROOT_DIR / "config" / "correlation_universes.yaml"
CACHE_DIR = ROOT_DIR / "data" / "cache" / "prices"

# A daily close is final once its market shuts. 18 hours keeps an evening fetch
# fresh for a run the next morning, while a run the following evening refetches
# and picks up that day's close.
CACHE_MAX_AGE_HOURS = 18

# Fetch the full history once and slice locally: stress windows reach back to
# 2008, and fetch_price_history only accepts a period, not start/end dates.
FETCH_PERIOD = "max"

# Robust z-score (median / MAD) above which a return is flagged. Six robust
# standard deviations is rare enough under fat tails to be worth a look (a
# data error, a split the adjustment missed) without flagging every crash day.
OUTLIER_Z = 6.0
# MAD of a normal distribution times this constant estimates its sigma.
MAD_TO_SIGMA = 1.4826

# Winsorizing clips each tail at this quantile. 0.5% touches only the most
# extreme handful of days per decade, so the result shows how much a few days
# drive the correlation without rewriting the distribution.
WINSOR_QUANTILE = 0.005

# Weekly returns are taken from the last common close of each week ending
# Friday, the convention most risk systems use.
WEEKLY_RULE = "W-FRI"

# Bad FX prints. A move beyond FX_SPIKE_Z robust standard deviations whose
# next-day move undoes all but FX_REVERSAL_SHARE of it is treated as a bad
# quote and the date dropped. Yahoo's EURUSD=X had five of these in 2008 (1.49
# on 2008-12-08 between 1.27 and 1.29), one reason FX now comes from FRED (see
# correlations/fred.py); the check stays as a guard on any FX source. Applied
# to FX only: a major currency pair does not jump 15% and return overnight,
# whereas a stock can genuinely spike and reverse (SPY on 2020-03-12/13).
FX_SPIKE_Z = 8.0
FX_REVERSAL_SHARE = 0.25

# Lags examined by fx_timing_check.
FX_TIMING_LAGS = (-1, 0, 1)
# Below this, the proxy and the FX rate are too weakly related for the check to
# say anything; the result is reported as inconclusive rather than as a pass
# or a failure.
FX_TIMING_MIN_SIGNAL = 0.05
# A neighbouring lag must beat the same-day correlation by this much to count
# as misdating. A fixing taken during the US session splits the day's
# information between two FX returns, so a near-tie is expected: the FRED noon
# rate gives SPY +0.080 at lag 0 and +0.084 at lag +1. Misdated data looks
# different: Yahoo's EURUSD=X gives +0.027 against +0.162 over the full
# sample, and -0.10 against +0.62 in 2011.
FX_TIMING_MARGIN = 0.10

FREQUENCIES = ("daily", "weekly")

Fetcher = Callable[[str, str], pd.DataFrame]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def load_correlation_config(path: Path = UNIVERSES_PATH) -> dict:
    """Load config/correlation_universes.yaml as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Correlation config not found at {path}.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping, got {type(data).__name__}.")
    return data


@dataclass(frozen=True)
class Universe:
    """A named set of symbols to correlate."""

    name: str
    symbols: tuple[str, ...]
    market_proxy: str
    groups: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    source: str = ""
    description: str = ""


# Peer groups carry no proxy of their own; the broad US market is the natural
# reference for regimes and the market factor.
DEFAULT_MARKET_PROXY = "SPY"


def resolve_universe(
    name: str,
    symbols_override: list[str] | None = None,
    config: dict | None = None,
) -> Universe:
    """Look a universe up by name: correlation config first, then peer groups.

    `symbols_override` replaces the symbol list while keeping the universe's
    proxy and group labels (unlabelled symbols are grouped as "other").
    """
    config = config if config is not None else load_correlation_config()
    universes = config.get("universes") or {}

    if name in universes:
        entry = universes[name] or {}
        symbols = [str(s).upper() for s in entry.get("symbols") or []]
        proxy = str(entry.get("market_proxy") or DEFAULT_MARKET_PROXY).upper()
        groups = {str(k).upper(): str(v) for k, v in (entry.get("groups") or {}).items()}
        source = "config/correlation_universes.yaml"
        description = str(entry.get("description") or "").strip()
    else:
        peer_groups = load_allowed_sources().get("peer_groups") or {}
        if name not in peer_groups:
            known = sorted(universes) + sorted(peer_groups)
            raise ValueError(
                f"Unknown universe {name!r}. Define it in config/correlation_universes.yaml "
                f"or use a peer group from config/allowed_sources.yaml. Known: {', '.join(known)}."
            )
        symbols = [str(s).upper() for s in peer_groups[name]]
        proxy = DEFAULT_MARKET_PROXY
        groups = {s: name for s in symbols}
        source = "config/allowed_sources.yaml peer_groups"
        description = f"Peer group {name!r}."

    if symbols_override:
        symbols = [s.strip().upper() for s in symbols_override if s.strip()]

    # Keep first occurrence order, drop duplicates.
    unique = tuple(dict.fromkeys(symbols))
    if len(unique) < 2:
        raise ValueError(f"Universe {name!r} needs at least two symbols, got {list(unique)}.")

    groups = {s: groups.get(s, "other") for s in unique}
    return Universe(
        name=name,
        symbols=unique,
        market_proxy=proxy,
        groups=MappingProxyType(groups),
        source=source,
        description=description,
    )


@dataclass(frozen=True)
class Listing:
    """Where a symbol trades and what it is quoted in."""

    symbol: str
    currency: str  # as quoted, e.g. "GBp" for London pence
    region: str


def _suffix(symbol: str) -> str:
    """Yahoo exchange suffix: ".PA" for "SU.PA", "" for "AAPL"."""
    if symbol.startswith("^"):
        return "^"  # indices never resolve by suffix; see symbol_overrides
    return "." + symbol.rsplit(".", 1)[1] if "." in symbol else ""


def listing_of(symbol: str, config: dict | None = None) -> Listing:
    """Currency and region for a symbol. Raises rather than guessing."""
    config = config if config is not None else load_correlation_config()
    symbol = symbol.strip().upper()

    overrides = {str(k).upper(): v for k, v in (config.get("symbol_overrides") or {}).items()}
    entry = overrides.get(symbol)
    if entry is None:
        suffix = _suffix(symbol)
        exchanges = config.get("exchanges") or {}
        entry = exchanges.get(suffix)
        if entry is None:
            raise ValueError(
                f"Unknown listing for {symbol!r} (suffix {suffix!r}): its currency cannot be "
                f"inferred. Add the suffix under `exchanges` or the symbol under "
                f"`symbol_overrides` in config/correlation_universes.yaml."
            )
    return Listing(symbol=symbol, currency=str(entry["currency"]), region=str(entry["region"]))


# --------------------------------------------------------------------------
# Fetching and cache
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceSeries:
    """One symbol's closing prices, indexed by local exchange date."""

    symbol: str
    close: pd.Series = field(repr=False)
    fetched_at: datetime | None
    from_cache: bool
    warnings: tuple[str, ...] = ()


def _cache_stem(symbol: str) -> str:
    """Filesystem-safe name: "^VIX" -> "_VIX", "EURUSD=X" -> "EURUSD_X"."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", symbol)


def to_exchange_dates(close: pd.Series) -> tuple[pd.Series, list[str]]:
    """Reduce a price series to one value per local exchange date.

    yfinance stamps each bar in the exchange's own timezone. Dropping the
    timezone keeps that local wall-clock date (a Paris close stays on its Paris
    date), which is what calendar alignment needs. Converting to UTC first
    would shift some bars onto the previous day.
    """
    notes: list[str] = []
    series = pd.Series(close, dtype=float).copy()
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    series.index = index.normalize()
    series = series.sort_index()

    duplicated = series.index.duplicated(keep="last")
    if duplicated.any():
        notes.append(f"{int(duplicated.sum())} duplicate date(s) collapsed to the last bar.")
        series = series[~duplicated]

    missing = series.isna()
    if missing.any():
        notes.append(f"{int(missing.sum())} missing close(s) dropped.")
        series = series[~missing]

    non_positive = series <= 0
    if non_positive.any():
        notes.append(
            f"{int(non_positive.sum())} non-positive close(s) dropped: a log return is "
            f"undefined there."
        )
        series = series[~non_positive]

    series.index.name = "date"
    return series, notes


def load_prices(
    symbol: str,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
    fetcher: Fetcher = fetch_price_history,
    now: datetime | None = None,
) -> PriceSeries:
    """Closing prices for one symbol, from the cache when fresh.

    A failed refetch falls back to a stale cached copy with a warning: stale
    prices are still real prices. With no cache at all, the error propagates.
    """
    symbol = symbol.strip().upper()
    now = now or datetime.now(timezone.utc)
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / f"{_cache_stem(symbol)}.csv"
    meta_path = cache_dir / f"{_cache_stem(symbol)}.meta.json"

    cached_at: datetime | None = None
    if csv_path.exists() and meta_path.exists():
        try:
            cached_at = datetime.fromisoformat(json.loads(meta_path.read_text())["fetched_at"])
        except (ValueError, KeyError, json.JSONDecodeError):
            cached_at = None

    fresh = cached_at is not None and now - cached_at <= timedelta(hours=CACHE_MAX_AGE_HOURS)
    if fresh and not refresh:
        return PriceSeries(symbol, _read_cache(csv_path), cached_at, from_cache=True)

    try:
        history = fetcher(symbol, FETCH_PERIOD)
    except Exception as exc:
        if cached_at is None:
            raise
        age = now - cached_at
        return PriceSeries(
            symbol,
            _read_cache(csv_path),
            cached_at,
            from_cache=True,
            warnings=(
                f"Refetch failed ({type(exc).__name__}: {exc}); using the cached copy from "
                f"{age.total_seconds() / 3600:.0f}h ago.",
            ),
        )

    close, notes = to_exchange_dates(history["Close"])
    if close.empty:
        raise ValueError(f"No usable closing prices for {symbol!r}.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Atomic: a half-written history must never be read back as a fresh one.
    atomic_write(csv_path, lambda p: close.to_frame("close").to_csv(p, index_label="date"))
    atomic_write_text(
        meta_path,
        json.dumps(
            {
                "symbol": symbol,
                "fetched_at": now.isoformat(),
                "rows": int(close.size),
                "first": close.index[0].date().isoformat(),
                "last": close.index[-1].date().isoformat(),
            },
            indent=2,
        )
    )
    return PriceSeries(symbol, close, now, from_cache=False, warnings=tuple(notes))


def _read_cache(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    series = frame["close"].astype(float)
    series.index = pd.DatetimeIndex(series.index)
    return series


# --------------------------------------------------------------------------
# Currency conversion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionLog:
    """What converting one symbol to the base currency cost."""

    symbol: str
    from_currency: str
    base: str
    fx_series: tuple[str, ...]  # recipe labels, e.g. "DEXUSEU"
    # Dates inside the FX history with no usable FX quote.
    dropped_dates: tuple[pd.Timestamp, ...] = ()
    # Asset dates before the FX history begins: not missing quotes, just no
    # conversion possible yet.
    before_fx_history: int = 0


def remove_reversal_spikes(
    series: pd.Series, z_threshold: float = FX_SPIKE_Z, share: float = FX_REVERSAL_SHARE
) -> tuple[pd.Series, tuple[pd.Timestamp, ...]]:
    """Drop single-day quotes that jump far out and come straight back.

    See FX_SPIKE_Z. Returns the cleaned series and the dropped dates.
    """
    returns = np.log(series).diff()
    median = float(returns.median())
    mad = float((returns - median).abs().median())
    if mad == 0.0 or len(returns) < 3:
        return series, ()
    z = (returns - median) / (MAD_TO_SIGMA * mad)
    following = returns.shift(-1)
    spike = (
        (z.abs() > z_threshold)
        & (np.sign(following) == -np.sign(returns))
        & ((returns + following).abs() < share * returns.abs())
    )
    bad = tuple(series.index[spike.fillna(False).to_numpy()])
    return series.drop(list(bad)), bad


@dataclass(frozen=True)
class FxTimingCheck:
    """Correlation of a market proxy's local return with the FX return at each lag.

    With correctly dated FX the strongest correlation sits at lag 0. If it sits
    elsewhere, the FX dates do not match the asset dates and the base-currency
    view is suspect. This is how the Yahoo FX dating problem was found; it now
    runs on every analysis as a guard.
    """

    proxy: str
    fx_symbol: str
    correlations: Mapping[int, float]
    n_observations: int
    aligned: bool
    note: str = ""


def fx_timing_check(
    proxy: str, proxy_close: pd.Series, fx_symbol: str, fx_close: pd.Series
) -> FxTimingCheck:
    """Measure which FX lag best matches the proxy's daily moves."""
    frame = pd.concat(
        {"proxy": np.log(proxy_close).diff(), "fx": np.log(fx_close).diff()}, axis=1, sort=True
    ).dropna()
    correlations = {
        k: float(frame["proxy"].corr(frame["fx"].shift(-k))) for k in FX_TIMING_LAGS
    }
    best = max(correlations, key=lambda k: abs(correlations[k]))
    if abs(correlations[best]) < FX_TIMING_MIN_SIGNAL:
        return FxTimingCheck(
            proxy, fx_symbol, MappingProxyType(correlations), len(frame), True,
            note=f"{proxy} and {fx_symbol} are too weakly related at every lag "
            f"(max |r| < {FX_TIMING_MIN_SIGNAL}) to verify the FX date convention.",
        )
    margin = abs(correlations[best]) - abs(correlations[0])
    aligned = best == 0 or margin < FX_TIMING_MARGIN
    if aligned:
        note = ""
    else:
        note = (
            f"{proxy} moves most with the {fx_symbol} return at lag {best:+d} "
            f"({correlations[best]:+.2f} against {correlations[0]:+.2f} same day): the FX "
            f"dates do not match the asset dates and base-currency figures are suspect."
        )
    return FxTimingCheck(
        proxy, fx_symbol, MappingProxyType(correlations), len(frame), aligned, note
    )


@dataclass(frozen=True)
class FxRecipe:
    """How to build "units of a currency per 1 EUR" from published series.

    FRED quotes each rate against the US dollar, so most per-EUR rates are a
    product or quotient: CHF per EUR = (CHF per USD) x (USD per EUR).
    """

    currency: str
    multiply: tuple[str, ...]
    divide: tuple[str, ...] = ()

    @property
    def series(self) -> tuple[str, ...]:
        return self.multiply + self.divide

    @property
    def label(self) -> str:
        text = " x ".join(self.multiply)
        return f"{text} / {' / '.join(self.divide)}" if self.divide else text


def required_fx(currencies: list[str], base: str, config: dict) -> dict[str, FxRecipe]:
    """Recipes for every currency needed to express `currencies` in `base`."""
    minor = config.get("minor_units") or {}
    per_eur = config.get("fx_per_eur") or {}
    majors = {str(minor.get(c, {}).get("major", c)) for c in currencies} | {base}
    needed: dict[str, FxRecipe] = {}
    for major in sorted(majors - {"EUR"}):
        entry = per_eur.get(major)
        if not entry or not entry.get("multiply"):
            raise ValueError(
                f"No FX series configured for {major}. Add it under `fx_per_eur` in "
                f"config/correlation_universes.yaml."
            )
        needed[major] = FxRecipe(
            currency=major,
            multiply=tuple(str(s) for s in entry["multiply"]),
            divide=tuple(str(s) for s in entry.get("divide") or ()),
        )
    return needed


def build_fx_per_eur(recipe: FxRecipe, components: Mapping[str, pd.Series]) -> pd.Series:
    """Combine published series into one per-EUR rate, on dates all of them have.

    A date missing from any component (a US bank holiday in the H.10 data, for
    example) is missing from the result rather than filled.
    """
    frame = pd.concat({s: components[s] for s in recipe.series}, axis=1, sort=True).dropna()
    rate = pd.Series(1.0, index=frame.index)
    for s in recipe.multiply:
        rate = rate * frame[s]
    for s in recipe.divide:
        rate = rate / frame[s]
    return rate.rename(f"{recipe.currency}_per_EUR")


def convert_to_base(
    close: pd.Series,
    symbol: str,
    currency: str,
    base: str,
    fx_per_eur: Mapping[str, pd.Series],
    fx_labels: Mapping[str, str],
    config: dict,
) -> tuple[pd.Series, ConversionLog]:
    """Express a price series in `base`, pivoting through EUR.

    `fx_per_eur[c]` is units of currency c per 1 EUR; `fx_labels[c]` names the
    series it was built from, for the log. The asset and each FX series are
    inner-joined on date: a day with no FX quote is dropped and logged rather
    than converted at a stale rate.
    """
    minor = (config.get("minor_units") or {}).get(currency)
    major = str(minor["major"]) if minor else currency
    divisor = float(minor["divisor"]) if minor else 1.0
    price = close / divisor

    if major == base:
        return price.rename(symbol), ConversionLog(symbol, currency, base, ())

    used: list[str] = []
    frame = price.to_frame("price")
    if major != "EUR":
        frame = frame.join(fx_per_eur[major].rename("local_per_eur"), how="left")
        used.append(fx_labels[major])
    if base != "EUR":
        frame = frame.join(fx_per_eur[base].rename("base_per_eur"), how="left")
        used.append(fx_labels[base])

    fx_start = max(
        fx_per_eur[c].index.min() for c in (major, base) if c != "EUR" and c in fx_per_eur
    )
    before = int((frame.index < fx_start).sum())
    frame = frame[frame.index >= fx_start]
    complete = frame.dropna()
    dropped = frame.index.difference(complete.index)
    converted = complete["price"]
    if "local_per_eur" in complete:
        converted = converted / complete["local_per_eur"]
    if "base_per_eur" in complete:
        converted = converted * complete["base_per_eur"]

    log = ConversionLog(
        symbol=symbol,
        from_currency=currency,
        base=base,
        fx_series=tuple(used),
        dropped_dates=tuple(pd.DatetimeIndex(dropped)),
        before_fx_history=before,
    )
    return converted.rename(symbol), log


# --------------------------------------------------------------------------
# Alignment and returns
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentLog:
    """How the common calendar was formed, per symbol."""

    first_dates: Mapping[str, pd.Timestamp]
    last_dates: Mapping[str, pd.Timestamp]
    limiting_symbol: str  # the series that starts last and so sets the start
    common_start: pd.Timestamp | None
    common_end: pd.Timestamp | None
    common_dates: int
    # Inside the common span: dates other markets traded but this one did not.
    missing_dates: Mapping[str, tuple[pd.Timestamp, ...]]
    # Inside the common span: this symbol's own dates removed because another
    # market was closed.
    dropped_dates: Mapping[str, tuple[pd.Timestamp, ...]]


def align_prices(closes: Mapping[str, pd.Series]) -> tuple[pd.DataFrame, AlignmentLog]:
    """Inner-join price levels on the dates every market traded.

    Joining prices (not returns) is what keeps each later return spanning the
    same interval for every asset; see the module docstring.
    """
    if len(closes) < 2:
        raise ValueError("Alignment needs at least two series.")

    first = {s: c.index.min() for s, c in closes.items()}
    last = {s: c.index.max() for s, c in closes.items()}
    start = max(first.values())
    end = min(last.values())
    limiting = max(first, key=lambda s: first[s])

    if start > end:
        empty = {s: () for s in closes}
        log = AlignmentLog(
            MappingProxyType(first), MappingProxyType(last), limiting, None, None, 0,
            MappingProxyType(empty), MappingProxyType(empty),
        )
        return pd.DataFrame(columns=list(closes)), log

    in_span = {s: c.loc[start:end] for s, c in closes.items()}
    union = pd.DatetimeIndex(sorted(set().union(*(c.index for c in in_span.values()))))
    frame = pd.concat(in_span, axis=1, join="inner").sort_index()
    common = frame.index

    missing = {s: tuple(union.difference(c.index)) for s, c in in_span.items()}
    dropped = {s: tuple(c.index.difference(common)) for s, c in in_span.items()}

    log = AlignmentLog(
        first_dates=MappingProxyType(first),
        last_dates=MappingProxyType(last),
        limiting_symbol=limiting,
        common_start=common.min() if len(common) else None,
        common_end=common.max() if len(common) else None,
        common_dates=int(len(common)),
        missing_dates=MappingProxyType(missing),
        dropped_dates=MappingProxyType(dropped),
    )
    return frame, log


def log_returns(prices: pd.DataFrame, frequency: str = "daily") -> pd.DataFrame:
    """Log returns of aligned prices, daily or weekly.

    Weekly returns run from the last common close of one week to the last
    common close of the next, so they inherit the alignment guarantee. Each is
    labelled with that actual last date, not the nominal Friday, so a week cut
    short by a holiday (or still in progress) does not carry a date that has
    not happened.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, got {frequency!r}.")
    if frequency == "weekly":
        prices = prices.dropna(how="any")
        weeks = prices.index.to_period(WEEKLY_RULE)
        prices = prices.groupby(weeks).tail(1)
    returns = np.log(prices.astype(float)).diff().iloc[1:]
    return returns.dropna(how="any")


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Outlier:
    symbol: str
    date: pd.Timestamp
    log_return: float
    robust_z: float


def find_outliers(
    returns: pd.DataFrame, threshold: float = OUTLIER_Z
) -> tuple[tuple[Outlier, ...], tuple[str, ...]]:
    """Flag returns more than `threshold` robust standard deviations out.

    Median and MAD rather than mean and standard deviation, because the very
    outliers being hunted inflate the standard deviation and hide themselves.
    """
    found: list[Outlier] = []
    notes: list[str] = []
    for symbol in returns.columns:
        column = returns[symbol].dropna()
        median = float(column.median())
        mad = float((column - median).abs().median())
        if mad == 0.0:
            notes.append(f"{symbol}: median absolute deviation is zero; outliers not assessed.")
            continue
        z = (column - median) / (MAD_TO_SIGMA * mad)
        for date in z.index[z.abs() > threshold]:
            found.append(Outlier(symbol, date, float(column[date]), float(z[date])))
    found.sort(key=lambda o: -abs(o.robust_z))
    return tuple(found), tuple(notes)


def winsorize(returns: pd.DataFrame, quantile: float = WINSOR_QUANTILE) -> pd.DataFrame:
    """Clip each column at its own `quantile` and `1 - quantile`."""
    lower = returns.quantile(quantile)
    upper = returns.quantile(1.0 - quantile)
    return returns.clip(lower=lower, upper=upper, axis=1)


def zero_return_counts(returns: pd.DataFrame) -> dict[str, int]:
    """Exact-zero returns per symbol.

    Alignment never manufactures these, so any that remain are in the source
    data: an illiquid day, or a stale quote worth checking.
    """
    return {s: int((returns[s] == 0.0).sum()) for s in returns.columns}
