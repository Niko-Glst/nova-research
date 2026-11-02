"""Command-line entry point: analyze one or more tickers.

    python research.py AAPL
    python research.py AAPL MSFT NVDA
    python research.py TSLA --period 6mo --insider-days 90
    python research.py AAPL --json > report.json

Research output only -- not financial advice, and no orders are placed.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

from agents import report_archive, research_report, signal_snapshots
from config.settings import ROOT_DIR


def _to_jsonable(value):
    """Convert dataclasses, mappings, arrays and datetimes into JSON-serializable structures.

    Walks the fields rather than using dataclasses.asdict, which deep-copies and
    fails on the read-only mappings the flow components carry.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):  # numpy values and arrays, pandas Series
        return _to_jsonable(value.tolist())
    if hasattr(value, "to_dict"):  # pandas DataFrame
        return _to_jsonable(value.to_dict(orient="list"))
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research.py",
        description="Run the six analysts on a ticker, with risk, position size and a verdict.",
        epilog="Research output only -- not financial advice.",
    )
    parser.add_argument("symbols", nargs="+", help="Ticker symbol(s), e.g. AAPL MSFT")
    parser.add_argument(
        "--period",
        default="1y",
        help="Price history window for technical analysis (default: 1y).",
    )
    parser.add_argument(
        "--insider-days",
        type=int,
        default=research_report.insider_analyst.DEFAULT_LOOKBACK_DAYS,
        help="Insider transaction lookback window in days (default: 180).",
    )
    parser.add_argument(
        "--no-peers",
        action="store_true",
        help="Skip the peer comparison (faster: avoids one request per peer).",
    )
    parser.add_argument(
        "--any-news-source",
        action="store_true",
        help="Count headlines from outside the allowlisted domains too.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--html",
        nargs="?",
        const=str(ROOT_DIR / "reports"),
        metavar="DIR",
        help="Write a full HTML report (charts, SWOT, thesis) into DIR "
        "(default: reports/) and open it in the browser.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run a backtest and Monte Carlo simulation, and chart them "
        "(refetches a 5-year price history).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="With --html, write the file but do not open a browser.",
    )
    return parser


def _write_html(report, directory: str) -> Path:
    """Render one report to HTML, archive a dated copy, and return its path."""
    return report_archive.save_report(report, directory)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    reports = []
    failed = False

    for symbol in args.symbols:
        try:
            report = research_report.analyze(
                symbol,
                period=args.period,
                include_peers=not args.no_peers,
                insider_lookback_days=args.insider_days,
                strict_allowlist=not args.any_news_source,
                run_simulation=args.simulate,
            )
        except Exception as exc:
            failed = True
            print(f"{symbol}: failed -- {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        reports.append(report)
        try:  # build the point-in-time signal history; never fail the run over it
            signal_snapshots.record(report)
        except Exception as exc:
            print(f"{symbol}: signal snapshot not saved -- {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        if not args.json:
            print(research_report.format_report(report))
            print()

        if args.html:
            path = _write_html(report, args.html)
            print(f"HTML report written to {path}")
            print(f"All saved reports: {Path(args.html) / report_archive.INDEX_FILE}")
            if not args.no_open:
                webbrowser.open(path.resolve().as_uri())

    if args.json:
        print(json.dumps([_to_jsonable(r) for r in reports], indent=2, default=str))

    return 1 if failed or not reports else 0


if __name__ == "__main__":
    from common.console import safe_console

    safe_console()
    raise SystemExit(main())
