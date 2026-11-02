"""Track record: every recorded verdict against what the stock then did.

Each run is stored with its verdict (agents/signal_snapshots.py). Here it is
settled the way TradingAgents settles its decisions (owner, 2026-10-28):

- **Entry**: the close of the first session dated after the day of the run (UTC).
  That close is always later than the run, so nothing known afterwards leaks in;
  at worst the entry is a day later than strictly possible.
- **Exit**: 1 week, 1, 3 and 6 months of the stock's own sessions after entry
  (config/scoring.yaml `track_record`). Until then the horizon is pending.
- **Benchmark**: the listing's local price index (agents/benchmarks.py), over the
  same dates. Excess return = stock return - index return. Price indices exclude
  dividends while the stock's prices include them, which flatters the stock by
  about 2-3% a year; SPY, the US benchmark, includes them.
- **Right or wrong**: a positive verdict is right when the stock beat its index,
  a negative one when it trailed. Mixed and inconclusive verdicts take no side
  and are shown without a mark.

Several runs of one ticker with the same entry close (three runs on a Sunday)
are one call: the last of them counts, so repeats do not inflate the record.

Shown only. Nothing here feeds back into a score or a weight.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from agents import benchmarks
from config.settings import ROOT_DIR, load_scoring_config

_CFG = load_scoring_config()["track_record"]
HORIZONS: dict[str, int] = {str(k): int(v) for k, v in _CFG["horizons"].items()}
OUTPUT_CSV = ROOT_DIR / str(_CFG["output_csv"])
OUTPUT_HTML = ROOT_DIR / str(_CFG["output_html"])

DIRECTION = {"positive": 1, "negative": -1}
HTML_MARKS = ("✓", "✗")  # check and cross; the terminal says right / wrong
DIVIDEND_NOTE = ("Local indices are price indices (no dividends) while the stock's prices "
                 "include dividends: about 2-3% a year in the stock's favour. SPY, the US "
                 "benchmark, includes dividends.")


@dataclass(frozen=True)
class Outcome:
    label: str
    sessions: int
    exit_date: date | None = None  # None: still pending
    stock_return: float = float("nan")
    benchmark_return: float = float("nan")

    @property
    def pending(self) -> bool:
        return self.exit_date is None

    @property
    def excess(self) -> float:
        return self.stock_return - self.benchmark_return


@dataclass(frozen=True)
class SettledCall:
    symbol: str
    recorded_at: str
    verdict: str
    score: float
    model_version: str
    benchmark: str | None
    entry_date: date | None = None
    entry_price: float = float("nan")
    outcomes: tuple[Outcome, ...] = ()
    note: str = ""

    def hit(self, outcome: Outcome) -> bool | None:
        """Right (True), wrong (False), or no call / not settled (None)."""
        direction = DIRECTION.get(self.verdict)
        if direction is None or outcome.pending or not math.isfinite(outcome.excess):
            return None
        return outcome.excess * direction > 0


def _run_day(recorded_at: str) -> date:
    stamp = datetime.fromisoformat(recorded_at)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).date()


def _price_on_or_before(close: pd.Series, day: pd.Timestamp) -> float:
    upto = close.loc[:day]
    return float(upto.iloc[-1]) if len(upto) else float("nan")


def _num(text) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value


def settle_row(row: Mapping[str, str], stock: pd.Series | None,
               index: pd.Series | None, benchmark: str | None) -> SettledCall:
    """One recorded run against the stock's and its index's closes."""
    base = dict(symbol=row.get("symbol", ""), recorded_at=row.get("recorded_at", ""),
                verdict=(row.get("verdict") or "").lower(), score=_num(row.get("score")),
                model_version=row.get("model_version") or "unversioned", benchmark=benchmark)
    pending = tuple(Outcome(label, n) for label, n in HORIZONS.items())
    if stock is None or stock.empty:
        return SettledCall(**base, outcomes=pending, note="No price history for the stock.")
    close = stock.dropna().sort_index()
    after = close.index[close.index > pd.Timestamp(_run_day(base["recorded_at"]))]
    if len(after) == 0:
        return SettledCall(**base, outcomes=pending, note="No close since the run yet.")
    entry_day = after[0]
    entry_pos = close.index.get_loc(entry_day)
    entry_price = float(close.iloc[entry_pos])
    bench_entry = (_price_on_or_before(index, entry_day)
                   if index is not None and not index.empty else float("nan"))
    outcomes = []
    for label, sessions in HORIZONS.items():
        exit_pos = entry_pos + sessions
        if exit_pos >= len(close):
            outcomes.append(Outcome(label, sessions))
            continue
        exit_day = close.index[exit_pos]
        bench_exit = (_price_on_or_before(index, exit_day)
                      if index is not None and not index.empty else float("nan"))
        outcomes.append(Outcome(label, sessions, exit_day.date(),
                                float(close.iloc[exit_pos]) / entry_price - 1.0,
                                bench_exit / bench_entry - 1.0))
    note = "" if benchmark else "No benchmark is configured for this market."
    return SettledCall(**base, entry_date=entry_day.date(), entry_price=entry_price,
                       outcomes=tuple(outcomes), note=note)


PriceLoader = Callable[[str], pd.Series]


def settle(rows: Iterable[Mapping[str, str]], prices: PriceLoader) -> list[SettledCall]:
    """Settle every run; one call per (ticker, entry close), the last run counting."""
    rows = [r for r in rows if r.get("symbol") and r.get("recorded_at")]
    cache: dict[str, pd.Series | None] = {}

    def load(symbol: str | None) -> pd.Series | None:
        if symbol is None:
            return None
        if symbol not in cache:
            try:
                cache[symbol] = prices(symbol)
            except Exception:
                cache[symbol] = None
        return cache[symbol]

    calls = []
    for row in sorted(rows, key=lambda r: r["recorded_at"]):
        benchmark = benchmarks.benchmark_for(row["symbol"])
        calls.append(settle_row(row, load(row["symbol"]), load(benchmark), benchmark))
    latest: dict[tuple, SettledCall] = {}
    for call in calls:  # later runs replace earlier ones with the same entry
        key = (call.symbol, call.entry_date or call.recorded_at)
        latest[key] = call
    return sorted(latest.values(), key=lambda c: c.recorded_at)


def ohlcv_closes(symbol: str) -> pd.Series:
    """Closes from the shared OHLCV cache (backtests/market_data.py)."""
    from backtests.market_data import load_ohlcv

    return load_ohlcv(symbol)["Close"]


def for_symbol(symbol: str, rows: Iterable[Mapping[str, str]] | None = None,
               prices: PriceLoader = ohlcv_closes) -> list[SettledCall]:
    """Earlier calls on one ticker; [] when it has never been recorded."""
    if rows is None:
        from agents import signal_snapshots

        rows = signal_snapshots.load_rows()
    mine = [r for r in rows if (r.get("symbol") or "").upper() == symbol.upper()]
    return settle(mine, prices) if mine else []


def summary(calls: Iterable[SettledCall]) -> pd.DataFrame:
    """Per verdict and horizon: calls settled, mean and median excess return, hit rate."""
    records = []
    for call in calls:
        for outcome in call.outcomes:
            if outcome.pending or not math.isfinite(outcome.excess):
                continue
            records.append({"verdict": call.verdict, "horizon": outcome.label,
                            "excess": outcome.excess, "hit": call.hit(outcome)})
    columns = ["verdict", "horizon", "calls", "mean_excess", "median_excess", "hit_rate"]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)
    order = {label: i for i, label in enumerate(HORIZONS)}
    out = []
    for (verdict, horizon), group in frame.groupby(["verdict", "horizon"]):
        hits = group["hit"].dropna()
        out.append({"verdict": verdict, "horizon": horizon, "calls": len(group),
                    "mean_excess": float(group["excess"].mean()),
                    "median_excess": float(group["excess"].median()),
                    "hit_rate": float(hits.astype(float).mean()) if len(hits) else float("nan")})
    table = pd.DataFrame(out, columns=columns)
    return table.sort_values(["verdict", "horizon"], key=lambda s: s.map(order)
                             if s.name == "horizon" else s).reset_index(drop=True)


def outcome_text(call: SettledCall, outcome: Outcome, marks: tuple[str, str] = ("right",
                                                                               "wrong")) -> str:
    """'+3.1% vs ^AEX (stock +5.0%), right' / 'pending'."""
    if outcome.pending:
        return "pending"
    hit = call.hit(outcome)
    mark = "" if hit is None else f", {marks[0] if hit else marks[1]}"
    if not math.isfinite(outcome.excess):
        return f"stock {outcome.stock_return:+.1%} (no benchmark){mark}"
    return (f"{outcome.excess:+.1%} vs {call.benchmark} (stock {outcome.stock_return:+.1%})"
            f"{mark}")


def to_frame(calls: Iterable[SettledCall]) -> pd.DataFrame:
    rows = []
    for call in calls:
        row = {"symbol": call.symbol, "recorded_at": call.recorded_at,
               "model_version": call.model_version, "verdict": call.verdict,
               "score": call.score, "benchmark": call.benchmark or "",
               "entry_date": call.entry_date.isoformat() if call.entry_date else "",
               "entry_price": call.entry_price, "note": call.note}
        for o in call.outcomes:
            row[f"excess_{o.label}"] = o.excess if not o.pending else float("nan")
            row[f"stock_{o.label}"] = o.stock_return if not o.pending else float("nan")
            row[f"hit_{o.label}"] = "" if call.hit(o) is None else str(call.hit(o)).lower()
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(calls: list[SettledCall], csv_path: Path | None = None,
                  html_path: Path | None = None) -> tuple[Path, Path]:
    from agents.html_report import _STYLE
    from common.files import atomic_write, atomic_write_text

    csv_path, html_path = csv_path or OUTPUT_CSV, html_path or OUTPUT_HTML
    atomic_write(csv_path, lambda p: to_frame(calls).to_csv(p, index=False))

    def pct(v):
        return "&ndash;" if not math.isfinite(v) else f"{v:+.1%}"

    table = summary(calls)
    summary_rows = "".join(
        f"<tr><td>{html.escape(r.verdict)}</td><td>{html.escape(r.horizon)}</td>"
        f"<td class='num'>{r.calls}</td><td class='num'>{pct(r.mean_excess)}</td>"
        f"<td class='num'>{pct(r.median_excess)}</td><td class='num'>"
        f"{'&ndash;' if not math.isfinite(r.hit_rate) else f'{r.hit_rate:.0%}'}</td></tr>"
        for r in table.itertuples())
    call_rows = "".join(
        f"<tr><td>{html.escape(c.symbol)}</td><td>{html.escape(c.recorded_at[:16])}</td>"
        f"<td>{html.escape(c.model_version)}</td><td>{html.escape(c.verdict)} "
        f"{'' if not math.isfinite(c.score) else f'{c.score:+.2f}'}</td>"
        f"<td>{c.entry_date or html.escape(c.note)}</td>"
        + "".join(f"<td>{html.escape(outcome_text(c, o, HTML_MARKS))}</td>"
                  for o in c.outcomes)
        + "</tr>" for c in reversed(calls))
    heads = "".join(f"<th>{html.escape(label)}</th>" for label in HORIZONS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Track record &middot; Nikolay Gelshtein</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<header><div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
<h1>Track record</h1>
<p class="muted-text">Every recorded verdict, settled against the stock's local index from the
first close after the run. {len(calls)} calls (repeat runs with the same entry count once).
Positive is right when the stock beat its index, negative when it trailed; mixed takes no
side. Calls from different model versions are pooled; each call's version is listed below.
Shown only: nothing here changes a score or a weight.</p></header>
<section class="card"><h2>By verdict</h2>
<table><thead><tr><th>Verdict</th><th>Horizon</th><th class="num">Settled calls</th>
<th class="num">Mean excess</th><th class="num">Median excess</th><th class="num">Right</th>
</tr></thead><tbody>{summary_rows or '<tr><td colspan="6">Nothing settled yet.</td></tr>'}
</tbody></table></section>
<section class="card"><h2>Every call</h2>
<table><thead><tr><th>Ticker</th><th>Run (UTC)</th><th>Model</th><th>Verdict</th><th>Entry</th>
{heads}</tr></thead><tbody>{call_rows}</tbody></table></section>
<section class="card"><h2>Read with care</h2><ul class="muted-text">
<li>{html.escape(DIVIDEND_NOTE)}</li>
<li>A handful of calls says little: a hit rate needs dozens of settled calls before it means
anything, and calls on the same days share the same market moves.</li>
</ul></section>
<footer>Research output. Not investment advice. No orders are placed.</footer>
</div></body></html>"""
    atomic_write_text(html_path, page)
    return csv_path, html_path
