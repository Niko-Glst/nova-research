"""Command-line entry point: correlation analysis on a universe of assets.

    python correlation.py --universe cross_asset
    python correlation.py --universe cross_asset --base USD --frequency weekly
    python correlation.py --universe semiconductors --winsorize
    python correlation.py --universe cross_asset --symbols SPY TLT GLD SU.PA
    python correlation.py --universe cross_asset --start 2015-01-01 --json > corr.json
    python correlation.py --universe global_multi_asset --html
    python correlation.py --pair AAPL MSFT --html          # one pair, with a forecast

Universes live in config/correlation_universes.yaml; any peer group from
config/allowed_sources.yaml also works by name. Prices are cached under
data/cache/prices/ (see correlations/data.py for the staleness rule).

Research output only -- not financial advice, and no orders are placed.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from correlations import data
from correlations.analysis import run_analysis
from correlations.report import format_analysis, to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="correlation.py",
        description="Correlation analysis across a universe of assets.",
        epilog="Research output only -- not financial advice.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--universe",
        help="Universe name from config/correlation_universes.yaml, or a peer group name.",
    )
    target.add_argument(
        "--pair", nargs=2, metavar=("SYMBOL_A", "SYMBOL_B"),
        help="Report on two assets over their full common history, with a forecast.",
    )
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL",
        help="Replace the universe's symbols (keeps its market proxy and labels).",
    )
    parser.add_argument(
        "--base", default="EUR", choices=["EUR", "USD"],
        help="Base currency for the converted view (default: EUR).",
    )
    parser.add_argument(
        "--frequency", default="both", choices=["daily", "weekly", "both"],
        help="Return frequency (default: both, which also enables the daily/weekly check).",
    )
    parser.add_argument("--start", help="First date to include, YYYY-MM-DD.")
    parser.add_argument("--end", help="Last date to include, YYYY-MM-DD.")
    parser.add_argument(
        "--winsorize", action="store_true",
        help="Also report estimates on returns clipped at 0.5%% per tail.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore the price cache and refetch everything."
    )
    parser.add_argument(
        "--no-evaluation", action="store_true",
        help="Skip the walk-forward evaluation of the estimators (about 10 s for 30 assets).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--html", nargs="?", const="correlation_reports", metavar="DIR",
        help="Also write the HTML report into DIR (default: correlation_reports/), archive a "
        "dated copy and open it in the browser.",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="With --html, write the file but do not open it."
    )
    return parser


def _run_pair(args: argparse.Namespace) -> int:
    from correlations.pair import analyze_pair, format_pair

    try:
        pair = analyze_pair(args.pair[0], args.pair[1], base=args.base, start=args.start,
                            end=args.end, refresh=args.refresh)
    except Exception as exc:
        print(f"Pair analysis failed -- {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(to_jsonable(pair), indent=2))
    else:
        print(format_pair(pair))
    if args.html:
        from correlations import html as html_report

        path = html_report.save_pair(pair, args.html)
        stream = sys.stderr if args.json else sys.stdout
        print(f"HTML report written to {path}", file=stream)
        print(f"All correlation reports: {Path(args.html) / html_report.INDEX_FILE}", file=stream)
        if not args.no_open:
            webbrowser.open(path.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pair:
        return _run_pair(args)
    frequencies = data.FREQUENCIES if args.frequency == "both" else (args.frequency,)
    try:
        analysis = run_analysis(
            args.universe,
            base=args.base,
            frequencies=frequencies,
            winsorize_returns=args.winsorize,
            refresh=args.refresh,
            start=args.start,
            end=args.end,
            symbols=args.symbols,
            evaluate=not args.no_evaluation,
        )
    except Exception as exc:
        print(f"Correlation analysis failed -- {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(to_jsonable(analysis), indent=2))
    else:
        print(format_analysis(analysis))

    if args.html:
        from correlations import html as html_report

        path = html_report.save(analysis, args.html)
        print(f"HTML report written to {path}", file=sys.stderr if args.json else sys.stdout)
        print(f"All correlation reports: {Path(args.html) / html_report.INDEX_FILE}",
              file=sys.stderr if args.json else sys.stdout)
        if not args.no_open:
            webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    from common.console import safe_console

    safe_console()
    sys.exit(main())
