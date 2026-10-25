"""Walk-forward test of signal weights: fit on the past, trade the future.

The only result that counts is out of sample. For each step:

1. Fit every method's weights on the previous TRAIN_DAYS of data (all stocks
   pooled, the same weights for every stock).
2. Trade the next STEP_DAYS with those weights, untouched.
3. Move forward STEP_DAYS and refit.

The out-of-sample periods are stitched into one track record per method and
compared with buy & hold of the same stocks and with the current pipeline
rule (the 50/200 crossover, which needs no fitting).

Two statistics guard against fooling ourselves:

- **Sharpe difference against buy & hold** (Jobson-Korkie with Memmel's 2003
  correction): is the gap larger than two correlated track records of this
  length would show by chance?
- **Deflated Sharpe ratio** (Bailey & Lopez de Prado 2014): the random search
  tries 10,000 weight/threshold combinations per refit, and the best of
  10,000 tries looks good even when nothing works. The DSR is the probability
  that the selected training Sharpe beats the best Sharpe expected from that
  many tries of pure noise, allowing for skew and fat tails.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtests.signal_weights import (
    FITTERS,
    TRADING_DAYS,
    Weights,
    evaluate,
    fixed,
    sharpe,
)

# Five years of training: long enough to contain more than one market regime,
# short enough to adapt when signal behaviour changes.
TRAIN_DAYS = 1260

# Refit every six months and trade the next six months out of sample.
STEP_DAYS = 126

# Euler-Mascheroni constant, used in the expected maximum of N normal draws.
_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class Panel:
    """Signals and returns for many stocks on one calendar."""

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    names: tuple[str, ...]  # signal names, the K axis
    x: np.ndarray = field(repr=False)  # (T, N, K) signals known at the previous close
    returns: np.ndarray = field(repr=False)  # (T, N) close-to-close simple returns
    coverage: Mapping[str, float] = field(default_factory=dict)  # share of non-missing


def build_panel(signals: Mapping[str, pd.DataFrame], closes: Mapping[str, pd.Series],
                start: str | None = None, end: str | None = None) -> Panel:
    """Align every stock's signals and returns on the union of trading days.

    Signals are lagged one day here, once, so no caller can forget to. A
    missing signal value (warm-up, no earnings report yet) becomes 0, "no
    view", and its coverage is reported.
    """
    symbols = tuple(signals)
    names = tuple(next(iter(signals.values())).columns)
    dates = pd.DatetimeIndex(sorted(set().union(*(c.index for c in closes.values()))))
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    t, n, k = len(dates), len(symbols), len(names)
    x = np.full((t, n, k), np.nan)
    r = np.full((t, n), np.nan)
    for j, s in enumerate(symbols):
        close = closes[s]
        ret = close.pct_change()
        lagged = signals[s][list(names)].shift(1)
        r[:, j] = ret.reindex(dates).to_numpy()
        x[:, j, :] = lagged.reindex(dates).to_numpy()
    traded = np.isfinite(r)
    coverage = {
        name: float(np.isfinite(x[:, :, i])[traded].mean()) if traded.any() else 0.0
        for i, name in enumerate(names)
    }
    return Panel(dates, symbols, names, np.nan_to_num(x, nan=0.0), r, coverage)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on erf: exact to 1e-10, no scipy."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _moments(daily: np.ndarray) -> tuple[float, float, float, int]:
    r = daily[np.isfinite(daily)]
    mean, std = r.mean(), r.std(ddof=1)
    z = (r - mean) / std
    return float(mean / std), float((z**3).mean()), float((z**4).mean()), int(r.size)


def deflated_sharpe(selected: np.ndarray, trial_sharpes_annual: np.ndarray) -> float:
    """Probability that the selected strategy's Sharpe beats the best of N noise tries.

    Bailey & Lopez de Prado (2014): SR0 = sqrt(V) * ((1 - g) Z(1 - 1/N) +
    g Z(1 - 1/(N e))), the expected maximum Sharpe of N independent tries with
    true Sharpe zero, then the probabilistic Sharpe ratio of the selected
    daily returns against SR0, corrected for skew and kurtosis. Sharpe ratios
    here are per day, as the formula requires.
    """
    trials = trial_sharpes_annual[np.isfinite(trial_sharpes_annual)] / math.sqrt(TRADING_DAYS)
    n_trials = trials.size
    if n_trials < 2:
        return float("nan")
    sr, skew, kurt, t = _moments(selected)
    sr0 = math.sqrt(trials.var(ddof=1)) * (
        (1 - _EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
        + _EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e))
    )
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0 or t < 3:
        return float("nan")
    return norm_cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom))


@dataclass(frozen=True)
class SharpeTest:
    difference: float  # annualised Sharpe of the strategy minus the benchmark's
    z: float
    p_value: float  # two-sided
    n_observations: int


def sharpe_difference_test(a: np.ndarray, b: np.ndarray) -> SharpeTest:
    """Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    t = a.size
    if t < 3 or a.std() == 0 or b.std() == 0:
        return SharpeTest(float("nan"), float("nan"), float("nan"), t)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = float(np.corrcoef(a, b)[0, 1])
    var = (2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / t
    if var <= 0:
        return SharpeTest(float("nan"), float("nan"), float("nan"), t)
    z = (sa - sb) / math.sqrt(var)
    return SharpeTest(float((sa - sb) * math.sqrt(TRADING_DAYS)), float(z),
                      float(math.erfc(abs(z) / math.sqrt(2.0))), t)


@dataclass(frozen=True)
class Performance:
    label: str
    annual_return_pct: float
    volatility_pct: float
    sharpe: float
    max_drawdown_pct: float
    total_return_pct: float
    time_in_market_pct: float
    trades_per_year: float  # entries per stock per year
    n_days: int


def performance(label: str, daily: np.ndarray, positions: np.ndarray | None = None) -> Performance:
    r = np.where(np.isfinite(daily), daily, 0.0)
    equity = np.cumprod(1.0 + r)
    years = max(len(r) / TRADING_DAYS, 1e-9)
    peak = np.maximum.accumulate(equity)
    in_market, trades = 100.0, 0.0
    if positions is not None and positions.size:
        in_market = float(np.nanmean(positions) * 100.0)
        entries = np.clip(np.diff(positions, axis=0, prepend=0.0), 0, None).sum()
        trades = float(entries / positions.shape[1] / years)
    return Performance(
        label=label,
        annual_return_pct=float((equity[-1] ** (1 / years) - 1) * 100.0) if equity[-1] > 0 else -100.0,
        volatility_pct=float(np.nanstd(daily, ddof=1) * math.sqrt(TRADING_DAYS) * 100.0),
        sharpe=sharpe(daily),
        max_drawdown_pct=float((equity / peak - 1.0).min() * 100.0),
        total_return_pct=float((equity[-1] - 1.0) * 100.0),
        time_in_market_pct=in_market,
        trades_per_year=trades,
        n_days=int(np.isfinite(daily).sum()),
    )


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Refit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    weights: Mapping[str, Weights]
    deflated_sharpe: float = float("nan")  # of the search winner on its training data


@dataclass(frozen=True)
class MethodResult:
    method: str
    daily: pd.Series = field(repr=False)  # stitched out-of-sample portfolio returns
    positions: np.ndarray = field(repr=False)
    performance: Performance = None  # type: ignore[assignment]
    versus_buy_hold: SharpeTest | None = None
    average_train_sharpe: float = float("nan")


@dataclass(frozen=True)
class WalkForwardResult:
    panel: Panel
    refits: tuple[Refit, ...]
    methods: Mapping[str, MethodResult]
    buy_hold: MethodResult
    crossover: MethodResult
    train_days: int
    step_days: int
    notes: tuple[str, ...] = ()

    @property
    def oos_start(self) -> pd.Timestamp | None:
        return self.refits[0].test_start if self.refits else None


CROSSOVER_RULE = {"trend": 1.0}


def walk_forward(
    panel: Panel,
    methods: Sequence[str] = tuple(FITTERS),
    train_days: int = TRAIN_DAYS,
    step_days: int = STEP_DAYS,
) -> WalkForwardResult:
    t, n, _ = panel.x.shape
    if t <= train_days + 20:
        raise ValueError(
            f"Only {t} days: need more than {train_days} for training plus a test period."
        )
    refits: list[Refit] = []
    oos: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    pos: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    carry: dict[str, np.ndarray | None] = {m: None for m in methods}
    train_sharpes: dict[str, list[float]] = {m: [] for m in methods}

    crossover_w = fixed(panel.names, CROSSOVER_RULE, 0.0, "crossover") \
        if "trend" in panel.names else None
    cross_daily, cross_pos, cross_carry = [], [], None

    start = train_days
    while start < t:
        stop = min(start + step_days, t)
        tr = slice(start - train_days, start)
        te = slice(start, stop)
        fitted: dict[str, Weights] = {}
        for m in methods:
            w = FITTERS[m](panel.x[tr], panel.returns[tr], panel.names)
            fitted[m] = w
            train_sharpes[m].append(w.train_sharpe)
            result = evaluate(panel.x[te], panel.returns[te], w, carry[m])
            oos[m].append(result.daily)
            pos[m].append(result.positions)
            carry[m] = result.positions[-1]
        if crossover_w is not None:
            result = evaluate(panel.x[te], panel.returns[te], crossover_w, cross_carry)
            cross_daily.append(result.daily)
            cross_pos.append(result.positions)
            cross_carry = result.positions[-1]

        dsr = float("nan")
        search = fitted.get("search")
        if search is not None and search.trial_sharpes is not None:
            selected = evaluate(panel.x[tr], panel.returns[tr], search).daily
            dsr = deflated_sharpe(selected, search.trial_sharpes)
        refits.append(Refit(panel.dates[tr.start], panel.dates[tr.stop - 1],
                            panel.dates[te.start], panel.dates[te.stop - 1], fitted, dsr))
        start = stop

    oos_dates = panel.dates[train_days:]
    with np.errstate(invalid="ignore"):
        bh_daily = np.nanmean(panel.returns[train_days:], axis=1)
    bh_positions = np.isfinite(panel.returns[train_days:]).astype(float)
    buy_hold = MethodResult("buy_hold", pd.Series(bh_daily, index=oos_dates), bh_positions,
                            performance("Buy & hold (equal weight)", bh_daily, None))

    def finish(label: str, daily_parts, pos_parts, train=None) -> MethodResult:
        daily = np.concatenate(daily_parts)
        positions = np.vstack(pos_parts)
        return MethodResult(
            label, pd.Series(daily, index=oos_dates), positions,
            performance(label, daily, positions),
            sharpe_difference_test(daily, bh_daily),
            float(np.nanmean(train)) if train else float("nan"),
        )

    results = {m: finish(m, oos[m], pos[m], train_sharpes[m]) for m in methods}
    crossover = finish("crossover", cross_daily, cross_pos) if cross_daily else None
    notes = []
    low = [f"{k} {v:.0%}" for k, v in panel.coverage.items() if v < 0.9]
    if low:
        notes.append("Signals missing on part of the sample (treated as 'no view'): "
                     + ", ".join(low) + ".")
    return WalkForwardResult(panel, tuple(refits), results, buy_hold, crossover,
                             train_days, step_days, tuple(notes))


def current_positions(panel: Panel, weights: Weights, latest_signals: Mapping[str, pd.Series]
                      ) -> dict[str, tuple[float, bool]]:
    """Today's score and long/cash decision per stock, from the latest signals."""
    out = {}
    for symbol, row in latest_signals.items():
        values = np.nan_to_num(row.reindex(list(panel.names)).to_numpy(dtype=float), nan=0.0)
        score = float(values @ weights.weights)
        out[symbol] = (score, score > weights.theta)
    return out
