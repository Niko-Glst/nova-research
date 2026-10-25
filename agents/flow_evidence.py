"""Flow, breadth and market-move readings as evidence lines for the thesis and SWOT.

Each line states a number, where it came from and its data flag, for example
"Net issuance +17.1% YoY (top decile among 20 us_large_caps stocks, source:
share count from SEC 10-K filed 2026-02-23; buyback line confirmed_zero)".

They are inputs, not verdicts (owner, 2026-10-22): they are listed next to the
thesis and the SWOT, and never change the thesis conclusion, its conditions or
the SWOT quadrants.

The rank compares the stock's net issuance with the latest point-in-time value
of every stock in the backfill universe (agents/flow_history.py). Without a
backfill the line says the rank is unavailable rather than guessing one.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import pandas as pd

from agents import breadth_analyst as ba
from agents import flow_analyst as fl
from agents import implied_move, snapshot_store
from config.settings import load_scoring_config

_CFG = load_scoring_config()
TOP_DECILE = float(_CFG["flows"]["evidence"]["top_decile"])
BOTTOM_DECILE = float(_CFG["flows"]["evidence"]["bottom_decile"])
MAX_FILING_AGE_DAYS = int(_CFG["history"]["max_filing_age_days"])
UNIVERSE = str(_CFG["history"]["universe"])


def issuance_reference(as_of: date | None = None) -> dict[str, float] | None:
    """Latest net issuance per ticker known by `as_of`, from the SEC backfill; None if absent."""
    as_of = as_of or date.today()
    try:
        long = snapshot_store.load(source=snapshot_store.SEC_PIT)
    except snapshot_store.StoreUnavailable:
        return None
    rows = long[long["field"] == "net_issuance"].copy()
    if rows.empty:
        return None
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    rows = rows[(rows["date"] <= as_of)
                & (rows["date"].map(lambda d: (as_of - d).days) <= MAX_FILING_AGE_DAYS)]
    latest = rows.sort_values(["date", "recorded_at"]).groupby("ticker").last()
    values = pd.to_numeric(latest["value"], errors="coerce").dropna()
    return values.to_dict() or None


def rank_text(value: float, reference: Mapping[str, float] | None, symbol: str = "") -> str:
    if not reference:
        return "rank unavailable: no backfill to compare with"
    others = [v for k, v in reference.items() if k != symbol and math.isfinite(v)]
    if not others:
        return "rank unavailable: no backfill to compare with"
    share = sum(v <= value for v in others) / len(others)
    where = ("top decile" if share >= TOP_DECILE else "bottom decile" if share <= BOTTOM_DECILE
             else f"{share:.0%} percentile")
    return f"{where} among {len(others)} {UNIVERSE} stocks"


def evidence_lines(flows: fl.FlowRead | None,
                   reference: Mapping[str, float] | None = None) -> list[str]:
    """Numbered, sourced and flagged readings; empty when there is no flow read."""
    if flows is None:
        return []
    lines = []
    issuance = flows.component("issuance")
    if issuance is not None:
        ni = issuance.values.get("net_issuance")
        source = issuance.reason.split("(", 1)[1].split(")", 1)[0] if "(" in issuance.reason \
            else "source not stated"
        if ni is not None and math.isfinite(ni):
            sec = "SEC 10-K" in source
            lines.append(
                f"Net issuance {ni:+.1%} YoY ({rank_text(ni, reference, flows.symbol)}, source: "
                f"share count from {source}{'' if sec else ', not confirmed by SEC'}; "
                f"buyback line {issuance.flag or 'unknown'})"
                + ("; left out of the score: the evidence conflicts"
                   if issuance.flag == fl.DATA_CONFLICT else ""))
        else:
            lines.append(f"Net issuance: not available ({issuance.flag or 'unknown'})")
    stock = flows.component("systematic_stock")
    if stock is not None and stock.score is not None:
        estimator = stock.flag.split(": ")[-1] if stock.flag else "unknown estimator"
        lines.append(
            f"Volatility {fl.STOCK_SHORT}-day/{fl.STOCK_LONG}-day ratio "
            f"{stock.values.get('vol_ratio', float('nan')):.2f} (z "
            f"{stock.values.get('z', float('nan')):+.2f}, {estimator}; part score "
            f"{stock.score:+.2f})")
    market = flows.component("systematic_market")
    if market is not None and market.score is not None:
        v = market.values
        lines.append(
            f"{fl.MARKET_PROXY} trend {v.get('return_1m', float('nan')):+.1%} / "
            f"{v.get('return_3m', float('nan')):+.1%} / {v.get('return_12m', float('nan')):+.1%} "
            f"(1/3/12 months), volatility ratio {v.get('vol_ratio', float('nan')):.2f}; "
            f"systematic-fund setting {market.score:+.0f}")
    if flows.options is not None and math.isfinite(flows.options.put_call_ratio):
        o = flows.options
        iv = (f", at-the-money implied vol {o.atm_implied_vol:.0%} vs {o.realised_vol:.0%} realised"
              if math.isfinite(o.atm_implied_vol) else "")
        lines.append(f"Put/call open interest {o.put_call_ratio:.2f} to {o.expirations[-1]}{iv} "
                     f"(not scored)")
    return lines


def breadth_line(breadth: ba.BreadthRead) -> str:
    """The breadth regime and the stock's group, as one numbered line."""
    r = breadth.regime
    if r is None:
        return f"{ba.EQUAL_WEIGHT}/{ba.CAP_WEIGHT}: not enough history for the range"
    score = "no breadth score" if breadth.score is None else f"leg score {breadth.score:+.0f}"
    return (f"{ba.EQUAL_WEIGHT}/{ba.CAP_WEIGHT} {r.ratio:.4f} at {r.position:.0%} of its "
            f"{ba.RANGE_DAYS}-day range ({r.low:.4f} to {r.high:.4f}), {ba.TURN_MA}-day average "
            f"{r.ma_turn:.4f}, {ba.TREND_DAYS}-day trend line {r.trend_change:+.1%}; regime "
            f"{r.regime:+.0f} ({r.state.replace('_', ' ')}); {breadth.group} stock, {score}")


def market_move_line(mm: implied_move.MarketMove) -> str:
    """The S&P 500's last move against the move VIX implied, as one line."""
    when = "so far " if mm.partial else ""
    recent = mm.recent_text().rstrip(".")
    return (f"S&P 500 {mm.move:+.2%} {when}on {mm.session} vs implied ±{mm.implied:.2%} "
            f"(VIX {mm.vol_prev:.2f} / {implied_move.DIVISOR:.2f}): {abs(mm.multiple):.2f}x, "
            f"{mm.tier}; {recent[:1].lower()}{recent[1:]} (shown, not scored)")
