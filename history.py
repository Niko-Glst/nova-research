"""Command-line entry point for the scoring history.

    python history.py rescore                  # rescore recorded runs under the current model
    python history.py backfill                 # point-in-time flow inputs into the snapshot store
    python history.py backtest                 # IC and quintile spreads from the stored backfill
    python history.py backtest --from-sources  # same, recomputing the backfill in memory
    python history.py breadth-backtest         # RSP/SPY regime against what followed
    python history.py implied-move-study       # S&P moves against VIX/16, and what followed

Research output only -- not financial advice, and no orders are placed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import yaml

from agents import breadth_analyst, breadth_history, flow_history, implied_move, rescore
from agents import sec_edgar, signal_snapshots, snapshot_store
from agents.research_report import MODEL_VERSION, WEIGHTS
from backtests.market_data import load_ohlcv
from config.settings import ROOT_DIR, load_scoring_config


def _rescore(args: argparse.Namespace) -> int:
    rows = signal_snapshots.load_rows()
    done = {(r["symbol"], r["recorded_at"]) for r in signal_snapshots.load_rescores()
            if r.get("model_version") == MODEL_VERSION}
    results, skipped = [], 0
    for row in rows:
        if (row.get("symbol"), row.get("recorded_at")) in done:
            skipped += 1
            continue
        results.append(rescore.rescore_row(row))
    written = signal_snapshots.record_rescores(results)
    missing = Counter(r.original_model for r in results if not r.rescored)
    print(f"Model {MODEL_VERSION}: {len(rows)} recorded run(s); {written} rescored now, "
          f"{skipped} already rescored, {sum(missing.values())} lack the inputs.")
    for original, count in sorted(missing.items()):
        print(f"  {count} run(s) under model {original}: no raw flow inputs "
              f"(recorded before 2.0.0); shown as a version break in the reports index.")
    for r in results:
        if r.rescored:
            print(f"  {r.symbol} {r.recorded_at}: {r.verdict} {r.score:+.2f} "
                  f"(was model {r.original_model})")
    return 0


def _universe() -> tuple[str, list[str]]:
    name = str(load_scoring_config()["history"]["universe"])
    with (ROOT_DIR / "config" / "strategy.yaml").open(encoding="utf-8") as fh:
        return name, list(yaml.safe_load(fh)["universes"][name]["symbols"])


def compute_backfill(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Long backfill rows for `symbols` plus the benchmark's OHLC, with notes."""
    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bench = load_ohlcv(flow_history.BENCHMARK)
    dates = flow_history.scoring_dates(bench.index)
    frames = [flow_history.market_backfill(bench["Close"], dates, recorded_at,
                                           flow_history.BENCHMARK)]
    notes = []
    for symbol in symbols:
        frames.append(flow_history.price_backfill(symbol, load_ohlcv(symbol), dates, recorded_at))
        try:
            cik = sec_edgar.cik_for(symbol)
            if cik is None:
                notes.append(f"{symbol}: no SEC CIK, so no issuance history.")
                continue
            frames.append(flow_history.sec_backfill(symbol, sec_edgar.company_facts(cik),
                                                    recorded_at))
        except Exception as exc:
            notes.append(f"{symbol}: SEC unavailable ({type(exc).__name__}: {exc}).")
    return pd.concat(frames, ignore_index=True), bench, notes


def _store_ready() -> bool:
    """The parquet store can be written; otherwise say how to fix it, before any work."""
    ok, reason = snapshot_store.pyarrow_status()
    if not ok:
        print(f"The snapshot store needs pyarrow ({reason}). To fix, run (from any folder): "
              f"{snapshot_store.PYARROW_FIX}", file=sys.stderr)
    return ok


def _backfill(args: argparse.Namespace) -> int:
    if not _store_ready():
        return 1
    name, symbols = _universe()
    long, _, notes = compute_backfill(symbols)
    path = snapshot_store.append(long)
    counts = long.groupby("source").size().to_dict()
    print(f"Backfilled {name} ({len(symbols)} stocks) into {path.name}: {counts}")
    for note in notes:
        print(f"  {note}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    name, symbols = _universe()
    if args.from_sources:
        long, bench, notes = compute_backfill(symbols)
        origin = "recomputed from prices and SEC filings in memory (not read from the store)"
    else:
        try:
            long = snapshot_store.load()
        except snapshot_store.StoreUnavailable as exc:
            print(f"{exc}\nOr run without the store: python history.py backtest "
                  f"--from-sources", file=sys.stderr)
            return 1
        long = long[long["source"].isin([snapshot_store.PRICE_BACKFILL, snapshot_store.SEC_PIT])]
        if long.empty:
            print("No backfill in the snapshot store. Run: python history.py backfill",
                  file=sys.stderr)
            return 1
        bench, notes = load_ohlcv(flow_history.BENCHMARK), []
        origin = "read from the snapshot store"
    long = (long.sort_values("recorded_at")
            .drop_duplicates(["date", "ticker", "field", "source"], keep="last"))
    closes = {s: load_ohlcv(s)["Close"] for s in symbols}
    dates = flow_history.scoring_dates(bench.index)
    panel = flow_history.build_panel(long, closes, bench["Close"], dates)
    ic_rows = flow_history.information_coefficients(panel)
    market = flow_history.market_table(panel, bench["Close"])
    meta = {"model_version": MODEL_VERSION, "universe": name, "tickers": len(symbols),
            "benchmark": flow_history.BENCHMARK, "first": str(dates.min().date()),
            "last": str(dates.max().date()),
            "notes": [f"Inputs {origin}."] + flow_history.caveats(len(symbols)) + notes}
    csv_path, html_path = flow_history.write_outputs(ic_rows, market, meta)
    print(f"Flow backtest, model {MODEL_VERSION}, {name}, {meta['first']} .. {meta['last']} "
          f"(inputs {origin})")
    print(f"  {'score':<10}{'h':>4}{'mean IC':>9}{'t':>7}{'IC>0':>6}{'months':>8}"
          f"{'top-bot':>9}{'t':>7}")
    for r in ic_rows:
        print(f"  {r.component:<10}{r.horizon_months:>3}m{r.mean_ic:>+9.3f}{r.ic_t:>+7.2f}"
              f"{r.hit_rate:>6.0%}{r.n_dates:>8}{r.spread:>+9.2%}{r.spread_t:>+7.2f}")
    print(f"CSV: {csv_path}\nHTML: {html_path}")
    return 0


def _breadth_backtest(args: argparse.Namespace) -> int:
    closes = {s: load_ohlcv(s)["Close"] for s in
              (breadth_analyst.EQUAL_WEIGHT, breadth_analyst.CAP_WEIGHT,
               *breadth_history.BREADTH_ETFS, *breadth_history.LEADER_ETFS)}
    ratio = breadth_analyst.ratio_series(closes[breadth_analyst.EQUAL_WEIGHT],
                                         closes[breadth_analyst.CAP_WEIGHT])
    dates = breadth_history.scoring_dates(ratio.index)
    panel = breadth_history.build_panel(
        ratio, {s: closes[s] for s in breadth_history.BREADTH_ETFS},
        {s: closes[s] for s in breadth_history.LEADER_ETFS}, dates)
    if panel.empty:
        print("Not enough RSP/SPY history for a single regime reading.", file=sys.stderr)
        return 1
    rows = breadth_history.summarize(panel)
    meta = {"model_version": MODEL_VERSION, "weight": WEIGHTS["breadth"],
            "first": str(panel.index.min().date()), "last": str(panel.index.max().date()),
            "notes": breadth_history.caveats(panel)}
    csv_path, html_path = breadth_history.write_outputs(rows, meta)
    print(f"Breadth backtest, model {MODEL_VERSION}, {meta['first']} .. {meta['last']}, "
          f"{len(panel)} month-ends")
    print(f"  states: {panel['state'].value_counts().to_dict()}")
    print(f"  {'outcome':<9}{'calls':<7}{'h':>4}{'months':>8}{'signed':>9}{'hit':>6}{'t':>7}"
          f"{'net':>9}{'t':>7}")
    for r in rows:
        if r.section == "call":
            print(f"  {r.outcome:<9}{r.label:<7}{r.horizon_months:>3}m{r.months:>8}"
                  f"{r.mean:>+9.2%}{r.hit_rate:>6.0%}{r.t:>+7.2f}{r.net_of_drift:>+9.2%}"
                  f"{r.net_t:>+7.2f}")
    print(f"CSV: {csv_path}\nHTML: {html_path}")
    return 0


def _implied_move_study(args: argparse.Namespace) -> int:
    table = implied_move.move_table(load_ohlcv(implied_move.INDEX)["Close"],
                                    load_ohlcv(implied_move.VOL_INDEX)["Close"])
    table = table.loc[implied_move.STUDY_START:]
    base = implied_move.base_rates(table)
    if base is None:
        print("Not enough S&P 500 and VIX history for the study.", file=sys.stderr)
        return 1
    tables = implied_move.study(table)
    csv_path, html_path = implied_move.write_study(tables, base)
    print(f"Implied move study, {base.first} .. {base.last}, {base.sessions:,} sessions; "
          f"actual moves averaged {base.realised_to_implied:.0%} of the implied move")
    for name, frame in tables.items():
        print(f"\n  {name}")
        print("  " + frame.to_string(index=False, float_format=lambda v: f"{v:.4f}")
              .replace("\n", "\n  "))
    print(f"\nCSV: {csv_path}\nHTML: {html_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="history.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rescore", help="Rescore recorded runs under the current model.")
    commands.add_parser("backfill", help="Point-in-time flow inputs into the snapshot store.")
    backtest = commands.add_parser("backtest", help="IC and quintile spreads of the flow score.")
    backtest.add_argument("--from-sources", action="store_true",
                          help="Recompute the backfill in memory instead of reading the store.")
    commands.add_parser("breadth-backtest",
                        help="The RSP/SPY regime against the ratio and sector ETFs afterwards.")
    commands.add_parser("implied-move-study",
                        help="S&P 500 moves against VIX/16 since 1990, and what followed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"rescore": _rescore, "backfill": _backfill, "backtest": _backtest,
                "breadth-backtest": _breadth_backtest,
                "implied-move-study": _implied_move_study}
    return commands[args.command](args)


if __name__ == "__main__":
    from common.console import safe_console

    safe_console()
    sys.exit(main())
