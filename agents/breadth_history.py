"""Breadth backtest: did the RSP/SPY regime point the right way afterwards?

At every month-end from the configured start, the regime is computed with the
live rule (breadth_analyst.regime_from_ratio) on the ratio up to that day only,
so each reading is point in time. It is then set against what followed over
1, 3 and 6 months:

- **ratio**: the change in ln(RSP/SPY). Did the market broaden (+) or
  narrow (-) as the regime said?
- **sectors**: the equal-weighted log return of the breadth-sector ETFs
  (XLV, XLF, XLP, XLU, XLI) minus that of the leaders' ETF (XLK). This is what
  the score means for a stock: +1 says breadth stocks beat the leaders.

For each regime, each state and each kind of call (all, zone, trend): months,
mean forward outcome and hit rate (the outcome's sign matches the call). The
signed outcome, regime x outcome over months with a call, gets a Newey-West
t-statistic (lag = horizon - 1 months, as the horizons overlap). Because the
ratio fell for most of the last decade, the signed outcome is also shown net
of that drift: a call that only ever said "narrowing" would look right without
knowing anything.

Nothing is fitted: the owner sets the leg's weight (0.8 at first; 0.5 since
2026-10-25, after this backtest found no significant edge).

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agents import breadth_analyst as ba
from agents.flow_history import newey_west_t
from config.settings import ROOT_DIR, load_scoring_config
from correlations.anchoring import month_ends

_CFG = load_scoring_config()["breadth"]
START = pd.Timestamp(_CFG["backtest_start"])
HORIZONS = tuple(int(h) for h in _CFG["backtest_horizons_months"])
BREADTH_ETFS = tuple(_CFG["backtest_sector_breadth"])
LEADER_ETFS = tuple(_CFG["backtest_sector_leaders"])
OUTPUT_CSV = ROOT_DIR / str(_CFG["output_csv"])
OUTPUT_HTML = ROOT_DIR / str(_CFG["output_html"])

OUTCOMES = {
    "ratio": f"ln {ba.EQUAL_WEIGHT}/{ba.CAP_WEIGHT} change",
    "sectors": f"breadth ETFs minus {'/'.join(LEADER_ETFS)}",
}
CALL_KINDS = {
    "all": ("bottom_turn", "top_turn", "trend_broadening", "trend_narrowing"),
    "zone": ("bottom_turn", "top_turn"),
    "trend": ("trend_broadening", "trend_narrowing"),
}
STATES = ("bottom_turn", "bottom_waiting", "trend_broadening", "trend_narrowing",
          "top_waiting", "top_turn")


def regimes(ratio: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The regime at each date, from the ratio up to that date only."""
    rows = []
    for day in dates:
        regime = ba.regime_from_ratio(ratio.loc[:day])
        if regime is None:
            continue
        rows.append({"date": day, "regime": regime.regime, "state": regime.state,
                     "position": regime.position, "trend_slope": regime.trend_slope})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _price_at(series: pd.Series, day: pd.Timestamp) -> float:
    upto = series.loc[:day]
    return float(upto.iloc[-1]) if len(upto) else float("nan")


def forward_log_change(series: pd.Series, day: pd.Timestamp, months: int) -> float:
    end = day + pd.DateOffset(months=months)
    if end > series.index.max():
        return float("nan")
    start, finish = _price_at(series, day), _price_at(series, end)
    return math.log(finish / start) if start > 0 and finish > 0 else float("nan")


def _basket(closes: Mapping[str, pd.Series], day: pd.Timestamp, months: int) -> float:
    values = [forward_log_change(c, day, months) for c in closes.values()]
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def build_panel(ratio: pd.Series, breadth_closes: Mapping[str, pd.Series],
                leader_closes: Mapping[str, pd.Series],
                dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per month-end: the regime then and the outcomes after it."""
    panel = regimes(ratio, dates)
    for h in HORIZONS:
        panel[f"ratio_{h}m"] = [forward_log_change(ratio, d, h) for d in panel.index]
        panel[f"sectors_{h}m"] = [_basket(breadth_closes, d, h) - _basket(leader_closes, d, h)
                                  for d in panel.index]
    return panel


@dataclass(frozen=True)
class Row:
    section: str  # "regime", "state" or "call"
    label: str
    outcome: str
    horizon_months: int
    months: int
    mean: float  # mean forward outcome (for "call": mean of regime x outcome)
    hit_rate: float  # outcome sign matches the call; NaN where there is no call
    t: float = float("nan")  # Newey-West, for "call" rows
    net_of_drift: float = float("nan")  # "call": regime x (outcome - mean over all months)
    net_t: float = float("nan")


def _hits(direction: pd.Series, outcome: pd.Series) -> float:
    called = direction != 0
    if not called.any():
        return float("nan")
    return float((np.sign(outcome[called]) == np.sign(direction[called])).mean())


def summarize(panel: pd.DataFrame) -> list[Row]:
    rows: list[Row] = []
    for outcome in OUTCOMES:
        for h in HORIZONS:
            column = f"{outcome}_{h}m"
            data = panel[["regime", "state", column]].dropna()
            y, regime = data[column], data["regime"]
            drift = float(y.mean()) if len(y) else float("nan")
            for value in (1.0, 0.0, -1.0):
                pick = regime == value
                rows.append(Row("regime", f"{value:+.0f}" if value else "0", outcome, h,
                                int(pick.sum()),
                                float(y[pick].mean()) if pick.any() else float("nan"),
                                _hits(regime[pick], y[pick])))
            for state in STATES:
                pick = data["state"] == state
                rows.append(Row("state", state, outcome, h, int(pick.sum()),
                                float(y[pick].mean()) if pick.any() else float("nan"),
                                _hits(regime[pick], y[pick])))
            for kind, states in CALL_KINDS.items():
                pick = data["state"].isin(states)
                signed = (regime[pick] * y[pick]).to_numpy()
                net = (regime[pick] * (y[pick] - drift)).to_numpy()
                rows.append(Row("call", kind, outcome, h, int(pick.sum()),
                                float(signed.mean()) if signed.size else float("nan"),
                                _hits(regime[pick], y[pick]), newey_west_t(signed, h - 1),
                                float(net.mean()) if net.size else float("nan"),
                                newey_west_t(net, h - 1)))
    return rows


def caveats(panel: pd.DataFrame) -> list[str]:
    counts = panel["state"].value_counts().to_dict() if not panel.empty else {}
    zone = sum(counts.get(s, 0) for s in CALL_KINDS["zone"])
    return [
        f"{len(panel)} month-ends; only {zone} of them were zone calls (a turn inside the top "
        f"or bottom {ba.ZONE:.0%}). Zone results rest on few, overlapping episodes.",
        "Months overlap at 3 and 6 months: the t-statistics allow for it, but the number of "
        "independent episodes is much smaller than the month count.",
        f"{'/'.join(LEADER_ETFS)} stands in for the mega-cap leaders; it also holds smaller "
        "tech names and leaves out Amazon, Alphabet and Meta after the 2018 sector reshuffle.",
        "Prices are dividend-adjusted closes (total return), as in the live leg.",
        "Net of drift subtracts the mean outcome over the whole sample, which is known only "
        "afterwards. It is a diagnostic, not something a live call could have used.",
        "No rule or weight was fitted to these results; they are a report.",
    ]


def write_outputs(rows: list[Row], meta: dict, csv_path: Path | None = None,
                  html_path: Path | None = None) -> tuple[Path, Path]:
    from agents.html_report import _STYLE

    csv_path, html_path = csv_path or OUTPUT_CSV, html_path or OUTPUT_HTML
    table = pd.DataFrame([r.__dict__ for r in rows])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    def fmt(v, spec):
        return "&ndash;" if v is None or not math.isfinite(v) else format(v, spec)

    def body(section: str, outcome: str) -> str:
        out = []
        for r in rows:
            if r.section != section or r.outcome != outcome:
                continue
            cells = [html.escape(r.label.replace("_", " ")), f"{r.horizon_months}m",
                     str(r.months), fmt(r.mean, "+.2%"), fmt(r.hit_rate, ".0%")]
            if section == "call":
                cells += [fmt(r.t, "+.2f"), fmt(r.net_of_drift, "+.2%"), fmt(r.net_t, "+.2f")]
            out.append("<tr>" + "".join(
                f"<td{'' if i == 0 else ' class=num'}>{c}</td>" for i, c in enumerate(cells))
                + "</tr>")
        return "".join(out)

    head = ("<th>{}</th><th class='num'>Horizon</th><th class='num'>Months</th>"
            "<th class='num'>{}</th><th class='num'>Hit rate</th>")
    sections = []
    for outcome, label in OUTCOMES.items():
        sections.append(f"""
<section class="card"><h2>{html.escape(label[:1].upper() + label[1:])}</h2>
<h3>Calls</h3>
<p class="muted-text">Mean of regime &times; outcome over months with a call (+1 or -1): positive
means the call pointed the right way on average. t: Newey-West, lag = horizon - 1 months.</p>
<table><thead><tr>{head.format('Calls', 'Signed mean')}<th class="num">t</th>
<th class="num">Net of drift</th><th class="num">t</th></tr></thead>
<tbody>{body('call', outcome)}</tbody></table>
<h3>By regime</h3>
<table><thead><tr>{head.format('Regime', 'Mean outcome')}</tr></thead>
<tbody>{body('regime', outcome)}</tbody></table>
<h3>By state</h3>
<table><thead><tr>{head.format('State', 'Mean outcome')}</tr></thead>
<tbody>{body('state', outcome)}</tbody></table></section>""")
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in meta.get("notes", []))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breadth backtest &middot; Nikolay Gelshtein</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<header><div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
<h1>Breadth backtest</h1>
<p class="muted-text">Model {html.escape(meta['model_version'])}. The {ba.EQUAL_WEIGHT}/{ba.CAP_WEIGHT}
regime at each month-end {html.escape(meta['first'])} to {html.escape(meta['last'])}, from
prices up to that day only, against what followed. Rule: {ba.RANGE_DAYS}-day range, top and
bottom {ba.ZONE:.0%} zones with a turn through the {ba.TURN_MA}-day average, otherwise the
{ba.TREND_DAYS}-day trend line. Breadth ETFs: {html.escape(', '.join(BREADTH_ETFS))}; leaders:
{html.escape(', '.join(LEADER_ETFS))}. Leg weight now {meta['weight']:.1f}; nothing is fitted
here.</p></header>
{''.join(sections)}
<section class="card"><h2>Read with care</h2><ul class="muted-text">{notes}</ul></section>
<footer>Research output. Not investment advice. No orders are placed.</footer>
</div></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return csv_path, html_path


def scoring_dates(calendar: pd.DatetimeIndex, start: pd.Timestamp = START) -> pd.DatetimeIndex:
    ends = month_ends(pd.DatetimeIndex(calendar))
    return ends[ends >= start]
