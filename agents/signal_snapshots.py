"""Record every analyst signal at the moment it is computed.

The signal-weight backtest (backtests/signals.py) can only test inputs with a
point-in-time history: what the signal said on a date, using only what was
known then. Sentiment, insider flow and valuation have no such history from
any free provider, so they cannot be weighted yet. This module starts building
that history: every research run appends one row per symbol, stamped with the
time it was computed.

The file is append-only, so a value cannot be revised after the fact; that is
what makes it point-in-time. The one exception is structural: when a new
version adds columns (always at the end), the file is widened once -- new
header, empty cells for the new columns in older rows -- and no existing value
changes. A header this version does not recognise is never appended to. It
lives in data/snapshots/ (gitignored, like all data).

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from agents import snapshot_store

ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT_DIR / "data" / "snapshots" / "signals.csv"

# Rescored results (agents/rescore.py) live next to the originals, append-only.
RESCORED_PATH = ROOT_DIR / "data" / "snapshots" / "rescored.csv"
RESCORED_COLUMNS = ("rescored_at", "symbol", "recorded_at", "original_model", "model_version",
                    "score", "verdict", "flows_signal")

# Column order is fixed: new columns go at the end, so older rows stay readable.
COLUMNS = (
    "recorded_at", "symbol", "verdict", "score", "confidence",
    "technical_signal", "fundamental_signal", "sentiment_signal", "insider_signal",
    "close", "sma_50", "sma_200", "rsi_14", "macd_histogram",
    "forward_pe", "trailing_pe", "revenue_growth", "profit_margin",
    "news_score", "news_articles", "reddit_score", "reddit_posts",
    "insider_buys", "insider_sells", "insider_net_value",
    # Added 2026-10-21 with agents/flow_analyst.py.
    "flows_signal", "net_buyback_yield", "market_up_horizons", "market_vol_ratio",
    "stock_vol_ratio", "put_call_oi_ratio", "atm_implied_vol", "institutions_pct",
    # Added 2026-10-22 with model 2.0.0 (net_buyback_yield stays empty from then on).
    "net_issuance", "buyback_flag", "vol_estimator", "stock_vol_z",
    "model_version", "market_return_1m", "market_return_3m", "market_return_12m",
    "asof", "schema_version",
    "breadth_signal", "breadth_group", "breadth_ratio", "breadth_position",
    "breadth_ma_turn", "breadth_trend_slope",
    # Added 2026-10-25 with agents/implied_move.py (shown, not scored).
    "market_move_session", "market_move", "market_vix_prev", "market_implied_move",
    "market_move_multiple", "market_move_tier", "market_move_partial",
)

# The wide CSV is schema 1; the long parquet store (agents/snapshot_store.py)
# is schema 2. After migration every new run goes to the long store.
CSV_SCHEMA_VERSION = "1"


def _num(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(number) else repr(number)


def snapshot_row(report, now: datetime | None = None) -> dict[str, str]:
    """Flatten a ResearchReport into one row of COLUMNS."""
    now = now or datetime.now(timezone.utc)
    row = {c: "" for c in COLUMNS}
    row.update(
        recorded_at=now.isoformat(timespec="seconds"),
        symbol=report.symbol,
        verdict=report.verdict,
        score=_num(report.score),
        confidence=_num(report.confidence),
        model_version=str(getattr(report, "model_version", "") or ""),
        asof=now.date().isoformat(),
    )
    if report.technical is not None:
        ind = report.technical.indicators
        row.update(technical_signal=report.technical.signal, close=_num(ind.get("close")),
                   sma_50=_num(ind.get("sma_50")), sma_200=_num(ind.get("sma_200")),
                   rsi_14=_num(ind.get("rsi_14")), macd_histogram=_num(ind.get("macd_histogram")))
    if report.fundamental is not None:
        m = report.fundamental.metrics
        row.update(fundamental_signal=report.fundamental.signal,
                   forward_pe=_num(m.get("forward_pe")), trailing_pe=_num(m.get("trailing_pe")),
                   revenue_growth=_num(m.get("revenue_growth")),
                   profit_margin=_num(m.get("profit_margin")))
    if report.sentiment is not None:
        s = report.sentiment
        row.update(sentiment_signal=s.signal, news_score=_num(s.news_score),
                   news_articles=str(s.news_sample), reddit_score=_num(s.reddit_score),
                   reddit_posts=str(s.reddit_sample))
    if report.insider is not None:
        i = report.insider
        row.update(insider_signal=i.signal, insider_buys=str(i.buy_count),
                   insider_sells=str(i.sell_count), insider_net_value=_num(i.net_value))
    breadth = getattr(report, "breadth", None)
    if breadth is not None:
        row.update(breadth_signal=breadth.signal, breadth_group=breadth.group)
        if breadth.regime is not None:
            r = breadth.regime
            row.update(breadth_ratio=_num(r.ratio), breadth_position=_num(r.position),
                       breadth_ma_turn=_num(r.ma_turn),
                       breadth_trend_slope=_num(r.trend_slope))
    mm = getattr(report, "market_move", None)
    if mm is not None:
        row.update(market_move_session=mm.session.isoformat(), market_move=_num(mm.move),
                   market_vix_prev=_num(mm.vol_prev), market_implied_move=_num(mm.implied),
                   market_move_multiple=_num(mm.multiple), market_move_tier=mm.tier,
                   market_move_partial=str(bool(mm.partial)).lower())
    flows = getattr(report, "flows", None)
    if flows is not None:
        def part(name: str, key: str) -> str:
            component = flows.component(name)
            return _num(component.values.get(key)) if component is not None else ""

        options = flows.options
        row.update(
            flows_signal=flows.signal,
            net_issuance=part("issuance", "net_issuance"),
            buyback_flag=(flows.component("issuance").flag
                          if flows.component("issuance") is not None else ""),
            vol_estimator=(flows.component("systematic_stock").flag.split(": ")[-1]
                           if flows.component("systematic_stock") is not None else ""),
            stock_vol_z=part("systematic_stock", "z"),
            market_return_1m=part("systematic_market", "return_1m"),
            market_return_3m=part("systematic_market", "return_3m"),
            market_return_12m=part("systematic_market", "return_12m"),
            market_up_horizons=part("systematic_market", "up_horizons"),
            market_vol_ratio=part("systematic_market", "vol_ratio"),
            stock_vol_ratio=part("systematic_stock", "vol_ratio"),
            put_call_oi_ratio=_num(options.put_call_ratio) if options else "",
            atm_implied_vol=_num(options.atm_implied_vol) if options else "",
            institutions_pct=_num(flows.institutions_pct),
        )
    return row


def _widen(path: Path, header: list[str]) -> None:
    """Rewrite with the current header, padding older rows; no value changes."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    temporary = path.with_suffix(".widening")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row + [""] * (len(COLUMNS) - len(row)))
    os.replace(temporary, path)


def record(report, path: Path | None = None, now: datetime | None = None,
           store_dir: Path | None = None) -> Path:
    """Append this report to the long store once migrated, else to the wide CSV.

    A store that cannot be written raises StoreUnavailable (an OSError) with the
    fix, which research.py and serve.py report without failing the run.
    """
    store_dir = Path(store_dir) if store_dir is not None else snapshot_store.STORE_DIR
    if snapshot_store.active(store_dir):
        row = snapshot_row(report, now)
        frame = snapshot_store.wide_to_long([row], snapshot_store.LIVE_RUN,
                                            lambda r: r.get("model_version", ""))
        return snapshot_store.append(frame, store_dir)
    path = Path(path) if path is not None else SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header != list(COLUMNS):
            if header != list(COLUMNS[: len(header)]):
                raise ValueError(
                    f"{path} has columns this version does not recognise; not appending."
                )
            _widen(path, header)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({**snapshot_row(report, now), "schema_version": CSV_SCHEMA_VERSION})
    return path


def load_rows(path: Path | None = None,
              store_dir: Path | None = None) -> list[dict[str, str]]:
    """Every recorded live run as a dict of strings (empty string = not recorded)."""
    store_dir = Path(store_dir) if store_dir is not None else snapshot_store.STORE_DIR
    if snapshot_store.active(store_dir):
        return snapshot_store.long_to_wide(
            snapshot_store.load(store_dir, snapshot_store.LIVE_RUN))
    path = Path(path) if path is not None else SNAPSHOT_PATH
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def record_rescores(results, path: Path | None = None, now: datetime | None = None,
                    store_dir: Path | None = None) -> int:
    """Append rescored runs (only those that could be rescored); never edits a row."""
    now = now or datetime.now(timezone.utc)
    store_dir = Path(store_dir) if store_dir is not None else snapshot_store.STORE_DIR
    rescored = [r for r in results if r.rescored]
    if snapshot_store.active(store_dir):
        if not rescored:
            return 0
        rows = [{"symbol": r.symbol, "recorded_at": r.recorded_at,
                 "asof": r.recorded_at[:10], "model_version": r.model_version,
                 "score": _num(r.score), "verdict": r.verdict, "flows_signal": r.flows_signal,
                 "original_model": r.original_model,
                 "rescored_at": now.isoformat(timespec="seconds")} for r in rescored]
        frame = snapshot_store.wide_to_long(rows, snapshot_store.RESCORE,
                                            lambda r: r["model_version"])
        snapshot_store.append(frame, store_dir)
        return len(rescored)
    path = Path(path) if path is not None else RESCORED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    written = 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESCORED_COLUMNS)
        if new_file:
            writer.writeheader()
        for r in results:
            if not r.rescored:
                continue
            writer.writerow({
                "rescored_at": now.isoformat(timespec="seconds"), "symbol": r.symbol,
                "recorded_at": r.recorded_at, "original_model": r.original_model,
                "model_version": r.model_version, "score": _num(r.score),
                "verdict": r.verdict, "flows_signal": r.flows_signal,
            })
            written += 1
    return written


def load_rescores(path: Path | None = None,
                  store_dir: Path | None = None) -> list[dict[str, str]]:
    store_dir = Path(store_dir) if store_dir is not None else snapshot_store.STORE_DIR
    if snapshot_store.active(store_dir):
        return snapshot_store.long_to_wide(
            snapshot_store.load(store_dir, snapshot_store.RESCORE))
    path = Path(path) if path is not None else RESCORED_PATH
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]
