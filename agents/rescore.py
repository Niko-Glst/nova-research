"""Rescore archived runs under the current scoring model, where their inputs allow it.

A snapshot row records each analyst's stance and, from model 2.0.0 on, the raw
flow inputs (net issuance and its buyback flag, the stock's volatility z and
ratio, the market's 1/3/12-month returns and volatility ratio). With those, a
run can be scored again under whatever model is current, using the very same
scoring functions as a live run (agents/flow_analyst.py), so a rescore of a
run made under the current model reproduces its score exactly.

A run recorded before those inputs existed cannot be rescored: the flows leg
would have to be guessed. It is reported with the inputs it lacks, and the
reports index shows a version break there instead of a comparison.

Rescored results are stored next to the original (agents/signal_snapshots.py),
never over it.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agents import breadth_analyst as ba
from agents import flow_analyst as fl
from agents import research_report

# Raw inputs the current flows leg needs, by part.
FLOW_INPUTS = {
    "issuance": ("net_issuance", "buyback_flag"),
    "systematic_market": ("market_return_1m", "market_return_3m", "market_return_12m",
                          "market_vol_ratio"),
    "systematic_stock": ("stock_vol_z", "stock_vol_ratio"),
}
LEG_SIGNALS = {"technical": "technical_signal", "fundamental": "fundamental_signal",
               "sentiment": "sentiment_signal", "insider": "insider_signal"}

# The fields above exist in snapshot rows from this model version on.
FIRST_RESCORABLE_MAJOR = 2


@dataclass(frozen=True)
class Rescore:
    symbol: str
    recorded_at: str
    original_model: str
    model_version: str  # the model it was rescored under
    score: float = float("nan")
    verdict: str = ""
    flows_signal: str = ""
    missing: tuple[str, ...] = ()  # inputs the run lacks; non-empty means not rescored

    @property
    def rescored(self) -> bool:
        return not self.missing


def _num(text: str | None) -> float | None:
    try:
        value = float(text) if text not in (None, "") else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except (ValueError, AttributeError):
        return 0


def flows_score(row: dict[str, str]) -> float:
    """The flows leg's weighted score from a row's raw inputs, current rules."""
    parts: dict[str, float | None] = {}
    ni = _num(row.get("net_issuance"))
    flag = row.get("buyback_flag") or fl.UNKNOWN
    parts["issuance"] = (None if ni is None or fl.is_data_conflict(ni, flag)
                         else fl.issuance_score(ni))
    returns = [_num(row.get(k)) for k in FLOW_INPUTS["systematic_market"][:3]]
    ratio = _num(row.get("market_vol_ratio"))
    parts["systematic_market"] = (fl.market_score(returns, ratio)
                                  if None not in returns and ratio is not None else None)
    z, stock_ratio = _num(row.get("stock_vol_z")), _num(row.get("stock_vol_ratio"))
    parts["systematic_stock"] = (fl.stock_score(z, stock_ratio)
                                 if z is not None and stock_ratio is not None else None)
    if not fl.has_stock_part(parts):  # the live rule: no stance from the market part alone
        return float("nan")
    return fl.weighted_score(parts)


def breadth_stance(row: dict[str, str]) -> str:
    """The breadth leg's stance from a row's regime inputs, current rules."""
    values = [_num(row.get(k)) for k in ("breadth_ratio", "breadth_position",
                                         "breadth_ma_turn", "breadth_trend_slope")]
    group = row.get("breadth_group") or ""
    if None in values or not group:
        return "unknown"
    regime = regime_from_inputs(*values)
    return ba.stance(ba.score_for(group, regime))


def regime_from_inputs(ratio: float, position: float, ma_turn: float,
                       trend_slope: float) -> float:
    """breadth_analyst.regime_from_ratio's rule, from the stored summary numbers."""
    if position <= ba.ZONE:
        return ba.ZONE_TURN_SCORE if ratio > ma_turn else ba.ZONE_WAITING_SCORE
    if position >= 1.0 - ba.ZONE:
        return -ba.ZONE_TURN_SCORE if ratio < ma_turn else -ba.ZONE_WAITING_SCORE
    return ba.TREND_SCORE if trend_slope > 0 else -ba.TREND_SCORE


def rescore_row(row: dict[str, str], model_version: str | None = None) -> Rescore:
    """Score one snapshot row under the current model, or say what it lacks."""
    model_version = model_version or research_report.MODEL_VERSION
    original = row.get("model_version") or ""
    base = dict(symbol=row.get("symbol", ""), recorded_at=row.get("recorded_at", ""),
                original_model=original or "unversioned", model_version=model_version)
    if _major(original) < FIRST_RESCORABLE_MAJOR:
        missing = tuple(f for fields in FLOW_INPUTS.values() for f in fields)
        return Rescore(**base, missing=missing)

    stances = {leg: row.get(column) or None for leg, column in LEG_SIGNALS.items()}
    flow = flows_score(row)
    stances["flows"] = fl.stance_from_score(flow)
    stances["breadth"] = breadth_stance(row)
    weighted, weight = 0.0, 0.0
    for leg, stance in stances.items():
        value = research_report._stance_value(stance)
        if value is None:
            continue
        weighted += value * research_report.WEIGHTS[leg]
        weight += research_report.WEIGHTS[leg]
    score = weighted / weight if weight else float("nan")
    if weight == 0:
        verdict = "inconclusive"
    elif score >= research_report._POSITIVE_THRESHOLD:
        verdict = "positive"
    elif score <= research_report._NEGATIVE_THRESHOLD:
        verdict = "negative"
    else:
        verdict = "mixed"
    return Rescore(**base, score=score, verdict=verdict, flows_signal=stances["flows"])
