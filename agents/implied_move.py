"""The S&P 500's daily move against the move VIX implied: the "VIX/16" rule.

VIX is the S&P 500's implied volatility over the next 30 days, per year. A year
has about 252 trading days and sqrt(252) = 15.87, so VIX/16 is the one-day move
(one standard deviation) the options market priced at the previous close: VIX
19.5 implies about 1.2% either way.

The owner's reading (2026-10-25): a move inside that is noise; a drop well
beyond it says the market's assumptions may have changed. Tiers (numbers in
config/scoring.yaml, judgement, checked against the data):

- within 1x the implied move: **noise**;
- 1x to 2x: **notable**, not yet a change;
- beyond 2x: **beyond**. Only a *drop* beyond 2x sets "assumptions may have
  changed": the owner's rule names drops, and in 1990-2026 big up days were not
  followed by more big days, while big drops were.

`history.py implied-move-study` reproduces the evidence: how often the S&P moved
beyond each multiple, by VIX level, and what followed each kind of day. VIX
overstates the typical move (the premium option sellers are paid), so moves
beyond 1x are rarer than a normal curve says.

VIX1D, CBOE's 1-day VIX, is shown alongside. With history only from 2023 it is
not used for the tiers.

Timing: the implied move is the VIX at the previous session's close (point in
time). VIX closes at 16:15 New York time, the S&P at 16:00. VIX prints on days
the stock market was shut are ignored. A report run during the session shows
the move so far today and says so.

Shown, not scored. Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agents.ta_analyst import fetch_price_history
from backtests.market_data import load_ohlcv
from config.settings import ROOT_DIR, load_scoring_config

_CFG = load_scoring_config()["implied_move"]
INDEX = str(_CFG["index"])
VOL_INDEX = str(_CFG["vol_index"])
NEXT_DAY_VOL_INDEX = str(_CFG["next_day_vol_index"])
TRADING_DAYS = int(_CFG["trading_days"])
DIVISOR = math.sqrt(TRADING_DAYS)
PRICE_PERIOD = str(_CFG["price_period"])
RECENT_SESSIONS = int(_CFG["recent_sessions"])
NOISE_MAX = float(_CFG["noise_max"])
NOTABLE_MAX = float(_CFG["notable_max"])
STUDY_START = pd.Timestamp(_CFG["study_start"])
FOLLOW_SESSIONS = int(_CFG["follow_sessions"])
FORWARD_SESSIONS = int(_CFG["forward_sessions"])
VIX_LEVELS = tuple(float(v) for v in _CFG["study_vix_levels"])
STUDY_MULTIPLES = tuple(float(v) for v in _CFG["study_multiples"])
OUTPUT_CSV = ROOT_DIR / str(_CFG["output_csv"])
OUTPUT_HTML = ROOT_DIR / str(_CFG["output_html"])

NOISE, NOTABLE, BEYOND, UNKNOWN = "noise", "notable", "beyond", "unknown"
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16

_TIER_TEXT = {
    NOISE: "within the implied move: noise",
    NOTABLE: f"beyond {NOISE_MAX:g}x but within {NOTABLE_MAX:g}x: notable, not yet a change",
    UNKNOWN: "no implied move to compare with",
}


def implied_daily_move(vol_index_close):
    """One-day, one-standard-deviation move implied by a vol index level (19.5 -> 0.0123)."""
    return vol_index_close / 100.0 / DIVISOR


def tier(multiple: float | None) -> str:
    if multiple is None or not math.isfinite(multiple):
        return UNKNOWN
    size = abs(multiple)
    return NOISE if size <= NOISE_MAX else NOTABLE if size <= NOTABLE_MAX else BEYOND


def normal_tail(k: float) -> float:
    """P(Z > k) for a standard normal variable."""
    return 0.5 * math.erfc(k / math.sqrt(2.0))


def _naive(series: pd.Series) -> pd.Series:
    out = series.dropna().astype(float).copy()
    index = pd.DatetimeIndex(out.index)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the exchange's own calendar date
    out.index = index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def move_table(index_close: pd.Series, vol_close: pd.Series) -> pd.DataFrame:
    """One row per index session: its move and the move implied at the previous close."""
    close = _naive(index_close)
    vol = _naive(vol_close).reindex(close.index)  # prints on market holidays drop out
    frame = pd.DataFrame({"close": close, "move": close.pct_change(), "vol": vol,
                          "vol_prev": vol.shift(1)})
    frame["implied"] = implied_daily_move(frame["vol_prev"])
    frame["multiple"] = frame["move"] / frame["implied"]
    frame = frame.dropna(subset=["move", "implied"])
    frame["tier"] = frame["multiple"].map(tier)
    return frame


# --------------------------------------------------------------------------
# What followed: shared by the report's base rates and the study
# --------------------------------------------------------------------------


def _ahead(values: np.ndarray, sessions: int, how: str) -> np.ndarray:
    """max / sum / std of the next `sessions` values after each position (NaN at the end)."""
    out = np.full(len(values), np.nan)
    if len(values) <= sessions:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values[1:], sessions)
    out[:len(windows)] = getattr(windows, how)(axis=1, **({"ddof": 1} if how == "std" else {}))
    return out


def follow_through(table: pd.DataFrame) -> pd.DataFrame:
    """Per session: what the next sessions brought, for comparing kinds of day."""
    big = (table["multiple"].abs() > NOTABLE_MAX).to_numpy(dtype=float)
    moves = table["move"].to_numpy(dtype=float)
    close = table["close"].to_numpy(dtype=float)
    forward = np.full(len(close), np.nan)
    forward[:-FORWARD_SESSIONS] = close[FORWARD_SESSIONS:] / close[:-FORWARD_SESSIONS] - 1.0
    realised = _ahead(moves, FORWARD_SESSIONS, "std") * DIVISOR * 100.0
    return pd.DataFrame({
        "big_soon": _ahead(big, FOLLOW_SESSIONS, "max"),
        "big_count": _ahead(big, FORWARD_SESSIONS, "sum"),
        "realised_to_vix": realised / table["vol"].to_numpy(dtype=float),
        "forward_return": forward,
    }, index=table.index)


@dataclass(frozen=True)
class BaseRates:
    """How often the S&P went beyond its implied move, and what followed big drops."""

    first: date
    last: date
    sessions: int
    down_beyond_noise: float  # share of sessions
    down_beyond_notable: float
    up_beyond_notable: float
    realised_to_implied: float  # RMS move over RMS implied move
    beyond_drops: int
    follow_after_beyond_drop: float  # P(a move beyond NOTABLE_MAX in the next sessions)
    follow_after_any: float


def base_rates(table: pd.DataFrame) -> BaseRates | None:
    if len(table) < FORWARD_SESSIONS * 2:
        return None
    m = table["multiple"]
    after = follow_through(table)["big_soon"]
    drops = m < -NOTABLE_MAX
    return BaseRates(
        first=table.index.min().date(), last=table.index.max().date(), sessions=len(table),
        down_beyond_noise=float((m < -NOISE_MAX).mean()),
        down_beyond_notable=float(drops.mean()),
        up_beyond_notable=float((m > NOTABLE_MAX).mean()),
        realised_to_implied=float(math.sqrt((table["move"] ** 2).mean()
                                            / (table["implied"] ** 2).mean())),
        beyond_drops=int(drops.sum()),
        follow_after_beyond_drop=float(after[drops].mean()) if drops.any() else float("nan"),
        follow_after_any=float(after.mean()),
    )


# --------------------------------------------------------------------------
# The reading for a report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketMove:
    session: date
    move: float
    vol_prev: float
    implied: float
    multiple: float
    tier: str
    partial: bool = False  # the last bar is the session so far, not a close
    next_day_vol_prev: float | None = None
    next_day_implied: float | None = None
    recent: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    base: BaseRates | None = None

    @property
    def direction(self) -> str:
        return "down" if self.move < 0 else "up" if self.move > 0 else "flat"

    @property
    def assumptions_changed(self) -> bool:
        """The owner's rule: a drop beyond NOTABLE_MAX implied moves."""
        return self.multiple < -NOTABLE_MAX

    @property
    def next_day_multiple(self) -> float | None:
        return self.move / self.next_day_implied if self.next_day_implied else None

    @property
    def recent_beyond_drops(self) -> list[date]:
        if self.recent.empty:
            return []
        hits = self.recent[self.recent["multiple"] < -NOTABLE_MAX]
        return [d.date() for d in hits.index]

    @property
    def tier_text(self) -> str:
        if self.tier != BEYOND:
            return _TIER_TEXT[self.tier]
        if self.assumptions_changed:
            return (f"a drop beyond {NOTABLE_MAX:g}x the implied move: the market's assumptions "
                    f"may have changed")
        return (f"an up-move beyond {NOTABLE_MAX:g}x: large, but only drops beyond "
                f"{NOTABLE_MAX:g}x mark a possible change")

    def describe(self) -> str:
        when = f"so far on {self.session}" if self.partial else f"on {self.session}"
        return (f"S&P 500 {self.move:+.2%} {when} against an implied move of "
                f"±{self.implied:.2%} (VIX {self.vol_prev:.2f} at the previous close / "
                f"{DIVISOR:.2f}): {abs(self.multiple):.2f}x, {self.tier_text}.")

    def recent_text(self) -> str:
        drops = self.recent_beyond_drops
        n = len(self.recent)
        if not drops:
            return f"No drop beyond {NOTABLE_MAX:g}x in the last {n} sessions."
        return (f"{len(drops)} drop(s) beyond {NOTABLE_MAX:g}x in the last {n} sessions, "
                f"latest {drops[-1]}.")

    def base_text(self) -> str | None:
        b = self.base
        if b is None:
            return None
        return (f"Since {b.first.year} ({b.sessions:,} sessions) the S&P fell beyond "
                f"{NOISE_MAX:g}x its implied move on {b.down_beyond_noise:.1%} of sessions and "
                f"beyond {NOTABLE_MAX:g}x on {b.down_beyond_notable:.1%}; after a drop beyond "
                f"{NOTABLE_MAX:g}x another move beyond it followed within {FOLLOW_SESSIONS} "
                f"sessions {b.follow_after_beyond_drop:.0%} of the time, against "
                f"{b.follow_after_any:.0%} after any session. Actual moves averaged "
                f"{b.realised_to_implied:.0%} of the implied move.")


def _is_partial(session: date, now: datetime | None) -> bool:
    stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    local = stamp.tz_convert(MARKET_TZ)
    return session == local.date() and local.hour < MARKET_CLOSE_HOUR


def reading(table: pd.DataFrame, next_day: pd.DataFrame | None = None,
            base: BaseRates | None = None, now: datetime | None = None) -> MarketMove:
    """The latest session of `table` as a MarketMove."""
    if table.empty:
        raise ValueError("no session with both an index move and an implied move")
    last = table.iloc[-1]
    day = table.index[-1]
    nd_vol = nd_implied = None
    if next_day is not None and day in next_day.index:
        nd_vol = float(next_day.loc[day, "vol_prev"])
        nd_implied = float(next_day.loc[day, "implied"])
    return MarketMove(
        session=day.date(), move=float(last["move"]), vol_prev=float(last["vol_prev"]),
        implied=float(last["implied"]), multiple=float(last["multiple"]), tier=str(last["tier"]),
        partial=_is_partial(day.date(), now), next_day_vol_prev=nd_vol,
        next_day_implied=nd_implied, recent=table.iloc[-RECENT_SESSIONS:].copy(), base=base)


def analyze(now: datetime | None = None) -> MarketMove:
    """Fetch the S&P 500 and VIX and read the last session against the implied move."""
    index = fetch_price_history(INDEX, period=PRICE_PERIOD)["Close"]
    table = move_table(index, fetch_price_history(VOL_INDEX, period=PRICE_PERIOD)["Close"])
    next_day = None
    try:  # shown alongside; its absence does not stop the reading
        next_day = move_table(index, fetch_price_history(NEXT_DAY_VOL_INDEX,
                                                         period=PRICE_PERIOD)["Close"])
    except Exception:
        next_day = None
    base = None
    try:  # cached full history; the reading stands without it
        history = move_table(load_ohlcv(INDEX)["Close"], load_ohlcv(VOL_INDEX)["Close"])
        base = base_rates(history.loc[STUDY_START:])
    except Exception:
        base = None
    return reading(table, next_day, base, now)


# --------------------------------------------------------------------------
# Study (history.py implied-move-study)
# --------------------------------------------------------------------------


def study(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The evidence behind the tiers, from a long move table."""
    m = table["multiple"]
    frequency = pd.DataFrame([{
        "multiple": k, "sessions_down_beyond": int((m < -k).sum()),
        "share_down_beyond": float((m < -k).mean()), "share_up_beyond": float((m > k).mean()),
        "normal_curve_per_side": normal_tail(k)} for k in STUDY_MULTIPLES])

    edges = (0.0, *VIX_LEVELS, math.inf)
    labels = [f"<{edges[1]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[1:-2], edges[2:-1])] \
        + [f"{edges[-2]:g}+"]
    buckets = pd.cut(table["vol_prev"], edges, labels=labels, right=False)
    by_vix = pd.DataFrame([{
        "vix_level": label, "sessions": int(len(g)), "share_of_sessions": len(g) / len(table),
        "share_down_beyond_noise": float((g["multiple"] < -NOISE_MAX).mean()),
        "share_down_beyond_notable": float((g["multiple"] < -NOTABLE_MAX).mean()),
        "std_multiple": float(g["multiple"].std())}
        for label, g in table.groupby(buckets, observed=True)])

    after = follow_through(table)
    top = max(STUDY_MULTIPLES)
    kinds = {
        "all sessions": pd.Series(True, index=table.index),
        f"down within {NOISE_MAX:g}x": (table["move"] < 0) & (m >= -NOISE_MAX),
        f"down {NOISE_MAX:g}x-{NOTABLE_MAX:g}x": (m < -NOISE_MAX) & (m >= -NOTABLE_MAX),
        f"down {NOTABLE_MAX:g}x-{top:g}x": (m < -NOTABLE_MAX) & (m >= -top),
        f"down beyond {top:g}x": m < -top,
        f"up beyond {NOTABLE_MAX:g}x": m > NOTABLE_MAX,
    }
    rows = []
    for name, mask in kinds.items():
        pick = mask & after["forward_return"].notna()
        a = after[pick]
        rows.append({
            "day": name, "sessions": int(pick.sum()),
            "p_big_move_next_sessions": float(a["big_soon"].mean()),
            "big_moves_next_month": float(a["big_count"].mean()),
            "realised_next_month_over_vix": float(a["realised_to_vix"].mean()),
            "forward_return": float(a["forward_return"].mean()),
            "share_forward_up": float((a["forward_return"] > 0).mean()),
        })
    return {"frequency": frequency, "by_vix": by_vix, "follow": pd.DataFrame(rows)}


def write_study(tables: dict[str, pd.DataFrame], base: BaseRates, csv_path: Path | None = None,
                html_path: Path | None = None) -> tuple[Path, Path]:
    from agents.html_report import _STYLE

    csv_path, html_path = csv_path or OUTPUT_CSV, html_path or OUTPUT_HTML
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        for i, (name, frame) in enumerate(tables.items()):
            if i:
                fh.write("\n")
            frame.assign(section=name).to_csv(fh, index=False)

    def pct(v, digits=1):
        return "&ndash;" if not math.isfinite(v) else f"{v:.{digits}%}"

    def table_html(frame: pd.DataFrame, columns: dict[str, tuple[str, object]]) -> str:
        head = "".join(f"<th{' class=num' if i else ''}>{html.escape(label)}</th>"
                       for i, (label, _) in enumerate(columns.values()))
        body = []
        for _, row in frame.iterrows():
            cells = []
            for i, (col, (_, fmt)) in enumerate(columns.items()):
                value = row[col]
                text = fmt(value) if callable(fmt) else html.escape(str(value))
                cells.append(f"<td{' class=num' if i else ''}>{text}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    frequency = table_html(tables["frequency"], {
        "multiple": ("Beyond", lambda v: f"{v:g}x implied"),
        "sessions_down_beyond": ("Sessions down", lambda v: f"{int(v):,}"),
        "share_down_beyond": ("Down", pct), "share_up_beyond": ("Up", pct),
        "normal_curve_per_side": ("Normal curve, per side", pct)})
    by_vix = table_html(tables["by_vix"], {
        "vix_level": ("VIX the day before", None),
        "sessions": ("Sessions", lambda v: f"{int(v):,}"),
        "share_of_sessions": ("Share", lambda v: pct(v, 0)),
        "share_down_beyond_noise": (f"Down beyond {NOISE_MAX:g}x", pct),
        "share_down_beyond_notable": (f"Down beyond {NOTABLE_MAX:g}x", pct),
        "std_multiple": ("Std of move / implied", lambda v: f"{v:.2f}")})
    follow = table_html(tables["follow"], {
        "day": ("Session", None), "sessions": ("Sessions", lambda v: f"{int(v):,}"),
        "p_big_move_next_sessions": (f"Move beyond {NOTABLE_MAX:g}x within {FOLLOW_SESSIONS}",
                                     lambda v: pct(v, 0)),
        "big_moves_next_month": (f"Such moves in next {FORWARD_SESSIONS}", lambda v: f"{v:.2f}"),
        "realised_next_month_over_vix": ("Next-month realised / VIX", lambda v: f"{v:.2f}"),
        "forward_return": (f"S&P next {FORWARD_SESSIONS}", lambda v: f"{v:+.2%}"),
        "share_forward_up": ("Up after", lambda v: pct(v, 0))})
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Implied move study &middot; Nikolay Gelshtein</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<header><div class="brand">NIKOLAY GELSHTEIN &middot; NOVA-RESEARCH</div>
<h1>Implied move study</h1>
<p class="muted-text">S&amp;P 500 close-to-close moves {base.first} to {base.last}
({base.sessions:,} sessions) against the move VIX implied at the previous close
(VIX / {DIVISOR:.2f}). Actual moves averaged {base.realised_to_implied:.0%} of the implied move.
Nothing here is fitted; the tiers ({NOISE_MAX:g}x, {NOTABLE_MAX:g}x) are the owner's.</p></header>
<section class="card"><h2>How often the S&amp;P went beyond its implied move</h2>{frequency}</section>
<section class="card"><h2>By VIX level</h2>{by_vix}</section>
<section class="card"><h2>What followed each kind of session</h2>
<p class="muted-text">Realised volatility over the next {FORWARD_SESSIONS} sessions, per year,
over the VIX at that session's close. Overlapping windows: big days cluster in a few episodes
(2008, 2020), so the rows for the largest drops rest on few independent events.</p>
{follow}</section>
<footer>Research output. Not investment advice. No orders are placed.</footer>
</div></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return csv_path, html_path
