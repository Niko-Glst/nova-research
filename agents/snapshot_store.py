"""Long-format, append-only snapshot store (schema 2), in parquet.

One row per (as-of date, ticker, field):

    date, ticker, field, value, source, filed_date, model_version,
    schema_version, recorded_at

- `date` is the as-of date: when the value was knowable (a live run's date, a
  price backfill's trading day, a 10-K's filing date).
- `value` is stored as text exactly as recorded, so nothing is rounded or
  reinterpreted on the way in or out; readers convert numbers themselves.
- `source`: live_run, sec_pit, price_backfill or rescore.
- `filed_date`: the SEC filing date for sec_pit rows, empty otherwise.
- `recorded_at`: when the row was written (UTC). It also groups one live run's
  fields, so two runs on the same day stay apart.

Append-only: every write adds a new part file under data/snapshots/long/;
existing parts are never opened for writing. A missing field is simply
absent, so a pivot back to wide gives NaN, never 0.

Parquet needs pyarrow (owner's choice, 2026-10-22). Without it the store says
so and how to install it, rather than falling back silently.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import ROOT_DIR, load_scoring_config

_CFG = load_scoring_config()
SCHEMA_VERSION = int(_CFG["schema_version"])
STORE_DIR = ROOT_DIR / str(_CFG["history"]["store_dir"])
MARKER_NAME = "MIGRATED_FROM_CSV"
# The interpreter actually running, by full path: a relative ".venv\..." only
# works from inside nova-research, and PowerShell elsewhere misreads it as a
# module name. The & makes PowerShell run a quoted path.
PYARROW_FIX = f'& "{sys.executable}" -m pip install pyarrow'

LONG_COLUMNS = ("date", "ticker", "field", "value", "source", "filed_date", "model_version",
                "schema_version", "recorded_at")
LIVE_RUN, SEC_PIT, PRICE_BACKFILL, RESCORE = "live_run", "sec_pit", "price_backfill", "rescore"

# Wide-row keys that become columns rather than fields.
_STRUCTURAL = {"recorded_at", "symbol", "asof", "schema_version", "model_version"}


class StoreUnavailable(OSError):
    """The parquet store cannot be used (pyarrow missing or blocked)."""


def pyarrow_status() -> tuple[bool, str]:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:  # ImportError, or a DLL blocked by Windows application control
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _require_pyarrow() -> None:
    ok, reason = pyarrow_status()
    if not ok:
        raise StoreUnavailable(f"The parquet snapshot store needs pyarrow ({reason}). "
                               f"To fix, run (from any folder): {PYARROW_FIX}")


def active(store_dir: Path = STORE_DIR) -> bool:
    """True once the CSV history has been migrated into this store."""
    return (Path(store_dir) / MARKER_NAME).exists()


def wide_to_long(
    rows: Iterable[dict[str, str]],
    source: str,
    model_version_of: Callable[[dict[str, str]], str],
    schema_version: int = SCHEMA_VERSION,
) -> pd.DataFrame:
    """Snapshot rows (dicts of strings) to long rows; empty cells produce no row."""
    records = []
    for row in rows:
        recorded_at = row.get("recorded_at", "")
        date = row.get("asof") or recorded_at[:10]
        version = model_version_of(row)
        for field, value in row.items():
            if field in _STRUCTURAL or value in (None, ""):
                continue
            records.append({
                "date": date, "ticker": row.get("symbol", ""), "field": field,
                "value": str(value), "source": source, "filed_date": "",
                "model_version": version, "schema_version": int(schema_version),
                "recorded_at": recorded_at,
            })
    return pd.DataFrame.from_records(records, columns=list(LONG_COLUMNS))


def long_to_wide(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Back to one dict per (ticker, recorded_at), in recorded order."""
    out = []
    keys = ["ticker", "recorded_at", "model_version"]
    for (ticker, recorded_at, version), group in frame.groupby(keys, sort=False):
        row = {"recorded_at": recorded_at, "symbol": ticker, "model_version": str(version)}
        row.update(dict(zip(group["field"], group["value"])))
        out.append(row)
    return sorted(out, key=lambda r: r["recorded_at"])


def append(frame: pd.DataFrame, store_dir: Path = STORE_DIR) -> Path:
    """Write `frame` as a new part file; never touches existing parts."""
    _require_pyarrow()
    missing = set(LONG_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Long rows are missing columns: {sorted(missing)}")
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = store_dir / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
    out = frame[list(LONG_COLUMNS)].astype({c: "string" for c in LONG_COLUMNS
                                            if c != "schema_version"})
    out["schema_version"] = out["schema_version"].astype("int64")
    out.to_parquet(path, engine="pyarrow", index=False)
    return path


def load(store_dir: Path = STORE_DIR, source: str | None = None) -> pd.DataFrame:
    """Every part, concatenated in write order (optionally one source)."""
    store_dir = Path(store_dir)
    parts = sorted(store_dir.glob("part-*.parquet")) if store_dir.exists() else []
    if not parts:
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    _require_pyarrow()
    frame = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in parts],
                      ignore_index=True)
    for column in LONG_COLUMNS:
        if column != "schema_version":
            frame[column] = frame[column].astype(object).where(frame[column].notna(), "")
    return frame if source is None else frame[frame["source"] == source].reset_index(drop=True)
