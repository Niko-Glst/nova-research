"""Command-line entry point: find signal weights and test them out of sample.

    python optimize.py --universe us_large_caps
    python optimize.py --universe us_large_caps --methods ridge equal
    python optimize.py --symbols AAPL MSFT JPM XOM --start 2010-01-01
    python optimize.py --universe us_large_caps --json > weights.json

Fits the weights of every backtestable signal (backtests/signals.py) on a
rolling training window, trades the following months with them untouched,
and compares the stitched out-of-sample record with buy & hold and with the
current pipeline rule (the 50/200-day crossover).

Research output only -- not financial advice, and no orders are placed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from backtests import market_data, signals
from backtests.signal_weights import FITTERS
from backtests.walk_forward import (
    STEP_DAYS,
    TRAIN_DAYS,
    MethodResult,
    WalkForwardResult,
    build_panel,
    current_positions,
    walk_forward,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "strategy.yaml"
WIDTH = 78

METHOD_LABELS = {
    "ridge": "Ridge regression",
    "max_sharpe": "Max-Sharpe (Sigma^-1 mu)",
    "search": "Random search",
    "equal": "Equal weights",
    "crossover": "Current rule: 50/200 crossover",
    "buy_hold": "Buy & hold",
}

CURRENT_STRATEGY = (
    "Your current pipeline trades one rule: long while the 50-day average is above the "
    "200-day average, cash otherwise, executed the next day with 10 bps per side. The "
    "analyst verdict (weights in config/scoring.yaml) is a report score and places no "
    "trades. In the weighted framework the crossover is the "
    "special case: weight 1 on 'trend', 0 on everything else, threshold 0."
)


def load_universe(name: str) -> tuple[list[str], str | None, str]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    universes = config.get("universes") or {}
    if name not in universes:
        raise ValueError(f"Unknown universe {name!r}; defined: {', '.join(universes)}.")
    entry = universes[name]
    return [s.upper() for s in entry["symbols"]], entry.get("start"), entry.get("description", "")


def load_signals(symbols: list[str], refresh: bool) -> tuple[dict, dict, dict, list[str]]:
    sigs, closes, latest, log = {}, {}, {}, []
    for s in symbols:
        try:
            ohlcv = market_data.load_ohlcv(s, refresh)
        except Exception as exc:
            log.append(f"{s}: prices unavailable ({type(exc).__name__}); left out.")
            continue
        try:
            earnings = market_data.load_earnings(s, refresh)
        except Exception as exc:
            earnings = None
            log.append(f"{s}: earnings unavailable ({type(exc).__name__}); surprise = no view.")
        frame = signals.compute_all(signals.SignalInputs(s, ohlcv, earnings))
        sigs[s], closes[s] = frame, ohlcv["Close"]
        latest[s] = frame.iloc[-1]
        n_reports = 0 if earnings is None else int(earnings["surprise_pct"].notna().sum())
        log.append(f"{s}: {len(ohlcv)} days {ohlcv.index.min().date()} .. "
                   f"{ohlcv.index.max().date()}, {n_reports} earnings reports")
    return sigs, closes, latest, log


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _f(value: float, spec: str = "+.2f") -> str:
    return "n/a" if value is None or not math.isfinite(value) else format(value, spec)


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _performance_rows(result: WalkForwardResult) -> list[str]:
    rows = [f"  {'':32s}{'Ann.ret':>8s}{'Vol':>7s}{'Sharpe':>8s}{'MaxDD':>8s}"
            f"{'InMkt':>7s}{'Trd/yr':>7s}{'p vs B&H':>10s}"]
    entries: list[MethodResult] = [result.buy_hold]
    if result.crossover is not None:
        entries.append(result.crossover)
    entries += list(result.methods.values())
    for m in entries:
        p = m.performance
        test = m.versus_buy_hold
        pv = "" if test is None else _f(test.p_value, ".3f")
        rows.append(
            f"  {METHOD_LABELS.get(m.method, m.method):32s}{_f(p.annual_return_pct, '7.1f')}%"
            f"{_f(p.volatility_pct, '6.1f')}%{_f(p.sharpe, '8.2f')}"
            f"{_f(p.max_drawdown_pct, '7.1f')}%{_f(p.time_in_market_pct, '6.0f')}%"
            f"{_f(p.trades_per_year, '7.1f')}{pv:>10s}"
        )
    return rows


def _weights_rows(result: WalkForwardResult) -> list[str]:
    last = result.refits[-1].weights
    names = result.panel.names
    head = f"  {'signal':22s}{'group':13s}" + "".join(f"{m[:10]:>11s}" for m in last)
    rows = [head]
    for i, name in enumerate(names):
        group = signals.REGISTRY[name].group
        rows.append(f"  {name:22s}{group:13s}"
                    + "".join(f"{_f(w.weights[i], '+11.2f')}" for w in last.values()))
    rows.append(f"  {'threshold':35s}" + "".join(f"{_f(w.theta, '+11.2f')}" for w in last.values()))
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(signals.REGISTRY[name].group, []).append(i)
    rows.append("  Share of absolute weight by group:")
    for g, idx in groups.items():
        rows.append(f"    {g:31s}" + "".join(
            f"{np.abs(w.weights[idx]).sum() / max(np.abs(w.weights).sum(), 1e-12):11.0%}"
            for w in last.values()))
    return rows


def _stability_rows(result: WalkForwardResult) -> list[str]:
    rows = ["  How much each method's weights moved between refits (average std per signal):"]
    for m in result.methods:
        stack = np.array([r.weights[m].weights for r in result.refits])
        rows.append(f"    {METHOD_LABELS[m]:30s} {np.mean(stack.std(axis=0)):.3f}")
    return rows


def _overfit_rows(result: WalkForwardResult) -> list[str]:
    rows = ["  Training Sharpe (what the optimiser saw) vs out-of-sample Sharpe (what it earned):"]
    for m, res in result.methods.items():
        rows.append(f"    {METHOD_LABELS[m]:30s} train {_f(res.average_train_sharpe)}   "
                    f"out of sample {_f(res.performance.sharpe)}")
    dsr = np.array([r.deflated_sharpe for r in result.refits])
    dsr = dsr[np.isfinite(dsr)]
    if dsr.size:
        rows.append(
            f"  Deflated Sharpe of the search winner: median {np.median(dsr):.2f} over "
            f"{dsr.size} refits ({(dsr > 0.95).mean():.0%} above 0.95). Below 0.95 the "
            f"winner is consistent with the luck of 10,000 tries."
        )
    return rows


def verdict(result: WalkForwardResult) -> list[str]:
    bh = result.buy_hold.performance
    best_name, best = max(result.methods.items(),
                          key=lambda kv: kv[1].performance.sharpe
                          if math.isfinite(kv[1].performance.sharpe) else -9)
    p = best.performance
    test = best.versus_buy_hold
    lines = []
    beats = p.sharpe > bh.sharpe
    significant = test is not None and math.isfinite(test.p_value) and test.p_value < 0.05
    lines.append(
        f"  Best out-of-sample method: {METHOD_LABELS[best_name]} with Sharpe {_f(p.sharpe)} "
        f"against {_f(bh.sharpe)} for buy & hold; annual return {_f(p.annual_return_pct, '.1f')}% "
        f"vs {_f(bh.annual_return_pct, '.1f')}%, worst drawdown "
        f"{_f(p.max_drawdown_pct, '.1f')}% vs {_f(bh.max_drawdown_pct, '.1f')}%."
    )
    if beats and significant:
        lines.append("  Its Sharpe is higher than buy & hold's and the difference is "
                     "statistically significant (p < 0.05).")
    elif beats:
        lines.append(f"  Its Sharpe is higher than buy & hold's, but the difference is not "
                     f"statistically significant (p = {_f(test.p_value, '.2f')}): it could be luck.")
    else:
        lines.append("  No fitted weighting beats buy & hold on a risk-adjusted basis out of sample.")
    cross = result.crossover
    if cross is not None and math.isfinite(cross.performance.sharpe):
        c = cross.performance
        relation = "above" if c.sharpe > bh.sharpe else "below"
        lines.append(
            f"  Your current rule (50/200 crossover, nothing fitted) has Sharpe {_f(c.sharpe)}, "
            f"{relation} buy & hold (p = {_f(cross.versus_buy_hold.p_value, '.2f')}), with a worst "
            f"drawdown of {_f(c.max_drawdown_pct, '.1f')}%."
        )
        if c.sharpe >= p.sharpe:
            lines.append("  No fitted weighting beats the simple rule out of sample: the "
                         "optimisation adds nothing over it.")
    lines.append("  Picking the best of several methods after the fact is itself a selection: "
                 "treat its edge over the others as an upper bound.")
    return lines


def format_result(result: WalkForwardResult, universe: str, description: str,
                  log: list[str], latest: dict) -> str:
    panel = result.panel
    lines = [_rule("="), f"  SIGNAL-WEIGHT BACKTEST: {universe}", _rule("=")]
    if description:
        lines.append(f"  {description}")
    lines += ["", "  HOW YOUR CURRENT STRATEGY WORKS", f"  {CURRENT_STRATEGY}", ""]
    lines.append(f"  Stocks ({len(panel.symbols)}): {', '.join(panel.symbols)}")
    lines.append(f"  Out of sample: {result.oos_start.date()} .. {panel.dates[-1].date()} "
                 f"({len(result.refits)} refits; train {result.train_days} days, trade "
                 f"{result.step_days} days, refit, repeat).")
    lines.append("  Signals in the backtest:")
    for name in panel.names:
        s = signals.REGISTRY[name]
        lines.append(f"    {name:22s} [{s.group}] {s.description} "
                     f"(coverage {panel.coverage[name]:.0%})")
    lines.append("  Registered but not backtestable yet:")
    for s in signals.unavailable():
        lines.append(f"    {s.name:22s} [{s.group}] {s.reason_unavailable}")
    for note in result.notes:
        lines.append(f"  Note: {note}")

    lines += ["", _rule(), "  OUT-OF-SAMPLE RESULTS (after 10 bps per trade)", _rule()]
    lines += _performance_rows(result)
    lines += ["", "  Verdict:"] + verdict(result)
    lines += ["", _rule(), "  LATEST WEIGHTS (fitted on the most recent training window)", _rule()]
    lines += _weights_rows(result)
    lines += ["", _rule(), "  OVERFITTING CHECKS", _rule()]
    lines += _overfit_rows(result)
    lines += _stability_rows(result)

    best_name = max(result.methods, key=lambda m: result.methods[m].performance.sharpe
                    if math.isfinite(result.methods[m].performance.sharpe) else -9)
    w = result.refits[-1].weights[best_name]
    lines += ["", _rule(), f"  WHAT {METHOD_LABELS[best_name].upper()} WOULD HOLD NOW", _rule()]
    for symbol, (score, long) in sorted(current_positions(panel, w, latest).items()):
        lines.append(f"    {symbol:6s} score {score:+.3f} vs threshold {w.theta:+.3f}  "
                     f"-> {'LONG' if long else 'cash'}")
    lines += ["", _rule(), "  DATA", _rule()] + [f"    {e}" for e in log]
    lines += ["", _rule(), "  LIMITATIONS", _rule(),
              "    - Survivorship: the stocks are today's large caps, chosen with hindsight.",
              "    - Sentiment, insider and valuation signals have no point-in-time history yet,",
              "      so their weights cannot be tested; they are being recorded from now on.",
              "    - Past out-of-sample results are still one history; regimes change.",
              "    - Costs are 10 bps per side; taxes, slippage on large orders and borrowing",
              "      are not modelled.",
              "", _rule("="), "  Research output only: not financial advice, no orders placed.",
              _rule("=")]
    return "\n".join(lines)


def to_json(result: WalkForwardResult) -> dict:
    def perf(m: MethodResult) -> dict:
        d = dict(m.performance.__dict__)
        if m.versus_buy_hold is not None:
            d["sharpe_difference_p_value"] = m.versus_buy_hold.p_value
        return d

    payload = {
        "symbols": list(result.panel.symbols),
        "signals": list(result.panel.names),
        "out_of_sample_start": str(result.oos_start.date()),
        "buy_hold": perf(result.buy_hold),
        "crossover": perf(result.crossover) if result.crossover else None,
        "methods": {m: perf(r) for m, r in result.methods.items()},
        "latest_weights": {m: {"weights": w.as_dict(), "theta": w.theta}
                           for m, w in result.refits[-1].weights.items()},
    }
    return json.loads(json.dumps(payload, default=lambda v: None if isinstance(v, float)
                                 and not math.isfinite(v) else str(v)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optimize.py", description=__doc__.splitlines()[0],
                                     epilog="Research output only -- not financial advice.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--universe", help="Universe name from config/strategy.yaml.")
    group.add_argument("--symbols", nargs="+", help="Explicit list of stock tickers.")
    parser.add_argument("--start", help="First date, YYYY-MM-DD (default: the universe's).")
    parser.add_argument("--end", help="Last date, YYYY-MM-DD.")
    parser.add_argument("--methods", nargs="+", choices=list(FITTERS), default=list(FITTERS))
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS)
    parser.add_argument("--refresh", action="store_true", help="Refetch instead of using the cache.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.universe:
        symbols, start, description = load_universe(args.universe)
        name = args.universe
    else:
        symbols, start, description = [s.upper() for s in args.symbols], None, ""
        name = "custom"
    start = args.start or start
    try:
        sigs, closes, latest, log = load_signals(symbols, args.refresh)
        if len(sigs) < 2:
            raise ValueError("Need at least two stocks with data.")
        panel = build_panel(sigs, closes, start, args.end)
        result = walk_forward(panel, args.methods, args.train_days, args.step_days)
    except Exception as exc:
        print(f"Backtest failed -- {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(to_json(result), indent=2))
    else:
        print(format_result(result, name, description, log, latest))
    return 0


if __name__ == "__main__":
    from common.console import safe_console

    safe_console()
    sys.exit(main())
