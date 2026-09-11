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
from dataclasses import asdict, is_dataclass

from agents import research_report


def _to_jsonable(value):
    """Convert dataclasses and datetimes into JSON-serializable structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research.py",
        description="Run technical, fundamental, sentiment and insider analysis on a ticker.",
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
    return parser


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
            )
        except Exception as exc:
            failed = True
            print(f"{symbol}: failed -- {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        reports.append(report)
        if not args.json:
            print(research_report.format_report(report))
            print()

    if args.json:
        print(json.dumps([_to_jsonable(r) for r in reports], indent=2))

    return 1 if failed or not reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
