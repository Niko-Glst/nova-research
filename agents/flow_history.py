"""Point-in-time backfill of the flow inputs, and a backtest of what they predict.

**Backfill** (history.py backfill), so the signal need not wait for live history:

- *price_backfill*: at every month-end, the stock's volatility reading
  (Yang-Zhang short window over close-to-close long window, z and ratio) and
  the market's 1/3/12-month returns and volatility ratio, from prices up to and
  including that day only. Fully reconstructible from price history.
- *sec_pit*: every 10-K in SEC companyfacts, dated by its **filing** date
  (`filed`), never its period end: net issuance from the filing's own two
  share counts, the year's repurchases and issuance, and the buyback flag
  (reported, or confirmed_zero when the 10-K has no repurchase fact).
  yfinance fundamentals are restated and undated, so they are never used here.

**Backtest** (history.py backtest), reported, not tuned:

- the information coefficient: each month-end, the Spearman rank correlation
  across stocks between a score and the forward 1, 3 and 6-month return in
  excess of the benchmark; averaged over months, with a Newey-West t-statistic
  (lag = horizon - 1 months, because overlapping horizons share returns);
- the top-minus-bottom quintile spread of forward excess returns;
- the same for each part: the stock's volatility, the issuance side (shares
  grew) and the buyback side (shares shrank) of net issuance;
- the market part is the same for every stock on a date, so it has no
  cross-sectional IC; a separate table shows the benchmark's forward return
  by market reading.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from agents import flow_analyst as fl
from agents import sec_edgar, snapshot_store
from agents import volatility as vol
from correlations.anchoring import month_ends
from agents.research_report import MODEL_VERSION
from config.settings import ROOT_DIR, load_scoring_config

_CFG = load_scoring_config()["history"]
BENCHMARK = str(_CFG["benchmark"])
START = pd.Timestamp(_CFG["start"])
HORIZONS = tuple(int(h) for h in _CFG["horizons_months"])
QUANTILES = int(_CFG["quantiles"])
MIN_TICKERS = int(_CFG["min_tickers_per_date"])
MAX_FILING_AGE_DAYS = int(_CFG["max_filing_age_days"])
OUTPUT_CSV = ROOT_DIR / str(_CFG["output_csv"])
OUTPUT_HTML = ROOT_DIR / str(_CFG["output_html"])

COMPONENTS = ("score", "vol", "issuance", "buyback")
COMPONENT_LABELS = {
    "score": "Flows score (all parts, configured weights)",
    "vol": "Stock volatility part",
    "issuance": "Issuance side (shares grew)",
    "buyback": "Buyback side (shares shrank)",
}


def _row(day, ticker, field, value, source, filed="") -> dict:
    return {"date": pd.Timestamp(day).date().isoformat(), "ticker": ticker, "field": field,
            "value": repr(float(value)) if isinstance(value, (float, np.floating)) else str(value),
            "source": source, "filed_date": filed, "model_version": MODEL_VERSION,
            "schema_version": snapshot_store.SCHEMA_VERSION, "recorded_at": ""}


# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------


def scoring_dates(calendar: pd.DatetimeIndex, start: pd.Timestamp = START) -> pd.DatetimeIndex:
    ends = month_ends(pd.DatetimeIndex(calendar))
    return ends[ends >= start]


def price_backfill(symbol: str, ohlc: pd.DataFrame, dates: pd.DatetimeIndex,
                   recorded_at: str) -> pd.DataFrame:
    """The stock's volatility inputs at each date, from prices up to that date."""
    rows = []
    for day in dates:
        history = ohlc.loc[:day]
        if len(history) <= fl.STOCK_LONG or history.index[-1] != day:
            continue
        reading = vol.volatility_ratio(history, fl.STOCK_SHORT, fl.STOCK_LONG)
        if not math.isfinite(reading.ratio):
            continue
        for field, value in (("stock_vol_ratio", reading.ratio), ("stock_vol_z", reading.z),
                             ("vol_estimator", reading.estimator)):
            rows.append(_row(day, symbol, field, value, snapshot_store.PRICE_BACKFILL))
    frame = pd.DataFrame(rows, columns=list(snapshot_store.LONG_COLUMNS))
    frame["recorded_at"] = recorded_at
    return frame


def market_backfill(close: pd.Series, dates: pd.DatetimeIndex, recorded_at: str,
                    symbol: str = fl.MARKET_PROXY) -> pd.DataFrame:
    """The market's trend and volatility inputs at each date (stored under its ticker)."""
    rows = []
    for day in dates:
        history = close.loc[:day]
        if history.empty or history.index[-1] != day:
            continue
        trend = fl.trend_returns(history)
        ratio = fl.volatility_ratio(history)
        if any(math.isnan(v) for v in trend.values()) or math.isnan(ratio):
            continue
        for field, value in (("market_return_1m", trend[21]), ("market_return_3m", trend[63]),
                             ("market_return_12m", trend[252]), ("market_vol_ratio", ratio)):
            rows.append(_row(day, symbol, field, value, snapshot_store.PRICE_BACKFILL))
    frame = pd.DataFrame(rows, columns=list(snapshot_store.LONG_COLUMNS))
    frame["recorded_at"] = recorded_at
    return frame


def sec_backfill(symbol: str, facts: dict, recorded_at: str) -> pd.DataFrame:
    """One set of rows per 10-K, dated by its filing date."""
    rows = []
    for report in sec_edgar.annual_reports(facts):
        filed = report.filed.isoformat()
        flag = fl.REPORTED if report.repurchased is not None else fl.CONFIRMED_ZERO
        values = {"net_issuance": report.net_issuance, "buyback_flag": flag,
                  "repurchased": report.repurchased, "issued": report.issued,
                  "fiscal_year_end": report.fiscal_year_end.isoformat(), "accn": report.accn}
        for field, value in values.items():
            if value is not None:
                rows.append(_row(report.filed, symbol, field, value, snapshot_store.SEC_PIT,
                                 filed))
    frame = pd.DataFrame(rows, columns=list(snapshot_store.LONG_COLUMNS))
    frame["recorded_at"] = recorded_at
    return frame


# --------------------------------------------------------------------------
# Panel: point-in-time scores and forward excess returns
# --------------------------------------------------------------------------


def _numeric(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    part = frame[frame["field"] == field][["date", "ticker", "value"]].copy()
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part["date"] = pd.to_datetime(part["date"])
    return part


def build_panel(long: pd.DataFrame, closes: dict[str, pd.Series], benchmark: pd.Series,
                dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (date, ticker): part scores known at that date, and forward returns."""
    prices = long[long["source"] == snapshot_store.PRICE_BACKFILL]
    sec = long[long["source"] == snapshot_store.SEC_PIT]
    z = _numeric(prices, "stock_vol_z").pivot_table(index="date", columns="ticker",
                                                    values="value", aggfunc="last")
    ratio = _numeric(prices, "stock_vol_ratio").pivot_table(index="date", columns="ticker",
                                                            values="value", aggfunc="last")
    market = {f: _numeric(prices[prices["ticker"] == fl.MARKET_PROXY], f).set_index("date")["value"]
              for f in ("market_return_1m", "market_return_3m", "market_return_12m",
                        "market_vol_ratio")}
    ni = _numeric(sec, "net_issuance")
    flags = sec[sec["field"] == "buyback_flag"][["date", "ticker", "value"]].copy()
    flags["date"] = pd.to_datetime(flags["date"])
    ni = ni.merge(flags.rename(columns={"value": "flag"}), on=["date", "ticker"], how="left")

    rows = []
    for day in dates:
        mkt = [market[f].get(day, np.nan) for f in market]
        market_part = (fl.market_score(mkt[:3], mkt[3])
                       if all(math.isfinite(v) for v in mkt) else None)
        for ticker in closes:
            if ticker == BENCHMARK:
                continue
            zt = z[ticker].get(day, np.nan) if ticker in z else np.nan
            rt = ratio[ticker].get(day, np.nan) if ticker in ratio else np.nan
            vol_part = fl.stock_score(zt, rt) if math.isfinite(zt) and math.isfinite(rt) else None
            known = ni[(ni["ticker"] == ticker) & (ni["date"] <= day)]
            issuance_part, issuance_side, buyback_side = None, np.nan, np.nan
            if not known.empty:
                latest = known.sort_values("date").iloc[-1]
                fresh = (day - latest["date"]).days <= MAX_FILING_AGE_DAYS
                if fresh and math.isfinite(latest["value"]) and \
                        not fl.is_data_conflict(latest["value"], str(latest["flag"])):
                    issuance_part = fl.issuance_score(latest["value"])
                    issuance_side = issuance_part if latest["value"] > 0 else 0.0
                    buyback_side = issuance_part if latest["value"] < 0 else 0.0
            score = fl.weighted_score({"issuance": issuance_part,
                                       "systematic_market": market_part,
                                       "systematic_stock": vol_part})
            row = {"date": day, "ticker": ticker, "score": score,
                   "vol": vol_part if vol_part is not None else np.nan,
                   "issuance": issuance_side, "buyback": buyback_side,
                   "market": market_part if market_part is not None else np.nan}
            close = closes[ticker]
            for h in HORIZONS:
                row[f"fwd_{h}m"] = _forward_excess(close, benchmark, day, h)
            rows.append(row)
    return pd.DataFrame(rows)


def _price_at(series: pd.Series, day: pd.Timestamp) -> float:
    upto = series.loc[:day]
    return float(upto.iloc[-1]) if len(upto) else float("nan")


def _forward_excess(close: pd.Series, benchmark: pd.Series, day: pd.Timestamp,
                    months: int) -> float:
    end = day + pd.DateOffset(months=months)
    if end > close.index.max() or end > benchmark.index.max():
        return float("nan")
    stock = _price_at(close, end) / _price_at(close, day) - 1.0
    bench = _price_at(benchmark, end) / _price_at(benchmark, day) - 1.0
    return stock - bench


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def newey_west_t(values: np.ndarray, lag: int) -> float:
    """Mean over its Newey-West standard error (Bartlett weights)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    centred = x - x.mean()
    variance = float(centred @ centred) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * float(centred[k:] @ centred[:-k]) / n
    return float(x.mean() / math.sqrt(variance / n)) if variance > 0 else float("nan")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    keep = x.notna() & y.notna()
    if keep.sum() < MIN_TICKERS or x[keep].nunique() < 2 or y[keep].nunique() < 2:
        return float("nan")
    return float(x[keep].rank().corr(y[keep].rank()))


def _spread(x: pd.Series, y: pd.Series) -> float:
    keep = x.notna() & y.notna()
    if keep.sum() < MIN_TICKERS or x[keep].nunique() < 2:
        return float("nan")
    buckets = pd.qcut(x[keep].rank(method="first"), QUANTILES, labels=False)
    returns = y[keep]
    return float(returns[buckets == QUANTILES - 1].mean() - returns[buckets == 0].mean())


@dataclass(frozen=True)
class IcRow:
    component: str
    horizon_months: int
    mean_ic: float
    ic_t: float
    hit_rate: float  # share of months with a positive IC
    n_dates: int
    spread: float  # mean top-minus-bottom quintile forward excess return
    spread_t: float


def information_coefficients(panel: pd.DataFrame) -> list[IcRow]:
    out = []
    for component in COMPONENTS:
        for h in HORIZONS:
            ics, spreads = [], []
            for _, group in panel.groupby("date"):
                ics.append(_spearman(group[component], group[f"fwd_{h}m"]))
                spreads.append(_spread(group[component], group[f"fwd_{h}m"]))
            ic = np.array(ics, dtype=float)
            sp = np.array(spreads, dtype=float)
            valid = ic[np.isfinite(ic)]
            out.append(IcRow(
                component, h,
                float(valid.mean()) if valid.size else float("nan"),
                newey_west_t(ic, h - 1),
                float((valid > 0).mean()) if valid.size else float("nan"),
                int(valid.size),
                float(np.nanmean(sp)) if np.isfinite(sp).any() else float("nan"),
                newey_west_t(sp, h - 1),
            ))
    return out


def market_table(panel: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Benchmark forward return by market reading (time series; one row per date)."""
    dates = panel.groupby("date")["market"].first().dropna()
    rows = []
    for h in HORIZONS:
        fwd = pd.Series({d: _price_at(benchmark, d + pd.DateOffset(months=h))
                         / _price_at(benchmark, d) - 1.0
                         for d in dates.index if d + pd.DateOffset(months=h) <= benchmark.index.max()})
        joined = pd.DataFrame({"market": dates, "fwd": fwd}).dropna()
        for reading, group in joined.groupby("market"):
            rows.append({"horizon_months": h, "market_reading": reading, "months": len(group),
                         "mean_benchmark_return": float(group["fwd"].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_outputs(ic_rows: list[IcRow], market: pd.DataFrame, meta: dict,
                  csv_path: Path = OUTPUT_CSV, html_path: Path = OUTPUT_HTML) -> tuple[Path, Path]:
    from agents.html_report import _STYLE

    table = pd.DataFrame([r.__dict__ for r in ic_rows])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.assign(section="cross_section_ic").to_csv(csv_path, index=False)
    if not market.empty:
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            fh.write("\n")
            market.assign(section="market_time_series").to_csv(fh, index=False)

    def fmt(v, spec):
        return "&ndash;" if v is None or not math.isfinite(v) else format(v, spec)

    ic_html = "".join(
        f"<tr><td>{html.escape(COMPONENT_LABELS[r.component])}</td><td class='num'>"
        f"{r.horizon_months}m</td><td class='num'>{fmt(r.mean_ic, '+.3f')}</td>"
        f"<td class='num'>{fmt(r.ic_t, '+.2f')}</td><td class='num'>{fmt(r.hit_rate, '.0%')}</td>"
        f"<td class='num'>{r.n_dates}</td><td class='num'>{fmt(r.spread, '+.2%')}</td>"
        f"<td class='num'>{fmt(r.spread_t, '+.2f')}</td></tr>" for r in ic_rows)
    market_html = "".join(
        f"<tr><td class='num'>{int(r.horizon_months)}m</td><td class='num'>"
        f"{r.market_reading:+.0f}</td><td class='num'>{int(r.months)}</td>"
        f"<td class='num'>{fmt(r.mean_benchmark_return, '+.2%')}</td></tr>"
        for r in market.itertuples()) if not market.empty else ""
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in meta.get("notes", []))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flow backtest &middot; Nikolay Gelshtein</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<header><div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
<h1>Flow backtest</h1>
<p class="muted-text">Model {html.escape(meta['model_version'])}, universe
{html.escape(meta['universe'])} ({meta['tickers']} stocks), month-ends
{html.escape(meta['first'])} to {html.escape(meta['last'])}, returns in excess of
{html.escape(meta['benchmark'])}. Point-in-time inputs only: prices up to each date, 10-K
facts from their filing date. Weights are not tuned on this.</p></header>
<section class="card"><h2>Information coefficient and quintile spread</h2>
<p class="muted-text">IC: Spearman rank correlation across stocks between the score and the
forward excess return, averaged over months. t: Newey-West, lag = horizon - 1 months
(overlapping horizons). Spread: top minus bottom quintile mean forward excess return.</p>
<table><thead><tr><th>Score</th><th class="num">Horizon</th><th class="num">Mean IC</th>
<th class="num">t</th><th class="num">IC &gt; 0</th><th class="num">Months</th>
<th class="num">Top &minus; bottom</th><th class="num">t</th></tr></thead>
<tbody>{ic_html}</tbody></table></section>
<section class="card"><h2>Market part (time series)</h2>
<p class="muted-text">The market reading is the same for every stock on a date, so it has no
cross-sectional IC. This shows the benchmark's own forward return by reading.</p>
<table><thead><tr><th class="num">Horizon</th><th class="num">Reading</th>
<th class="num">Months</th><th class="num">Benchmark forward return</th></tr></thead>
<tbody>{market_html}</tbody></table></section>
<section class="card"><h2>Read with care</h2><ul class="muted-text">{notes}</ul></section>
<footer>Research output. Not investment advice. No orders are placed.</footer>
</div></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return csv_path, html_path


def caveats(tickers: int) -> list[str]:
    return [
        f"{tickers} stocks chosen today (survivorship bias): the winners that stayed large.",
        f"Quintiles of about {max(tickers // QUANTILES, 1)} stocks are noisy; read the IC first.",
        "Overlapping 3- and 6-month horizons share returns; the t-statistics allow for that.",
        "10-K facts are dated by filing, but later restatements are not visible point in "
        "time: companyfacts keeps the values as filed in each document.",
        "No weights were fitted to these results (owner's rule); they are a report.",
    ]


def today_iso() -> str:
    return date.today().isoformat()
