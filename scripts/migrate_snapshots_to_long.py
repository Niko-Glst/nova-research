"""One-time migration: the wide snapshot CSV (schema 1) into the long parquet store (schema 2).

    .venv\\Scripts\\python.exe scripts/migrate_snapshots_to_long.py

1. Read every CSV row as text.
2. Convert to long rows; convert back; assert every recorded value is unchanged
   (in memory).
3. Write the long rows as one part file; read the store back; assert again.
4. Assert the CSV file itself is byte-for-byte unchanged (it stays as the
   original record), and write a marker so the migration can never run twice.

Rows written before model versioning get an inferred version: 1.1.0 when the
flows leg is present, 1.0.0 otherwise.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import report_archive, signal_snapshots, snapshot_store  # noqa: E402

# Wide-row keys that are not fields in the long store.
_NOT_FIELDS = {"recorded_at", "symbol", "asof", "schema_version", "model_version"}


def infer_model_version(row: dict[str, str]) -> str:
    if row.get("model_version"):
        return row["model_version"]
    return report_archive.INFERRED_FLOWS if row.get("flows_signal") else \
        report_archive.INFERRED_BASE


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def assert_round_trip(rows: list[dict[str, str]], back: list[dict[str, str]]) -> int:
    """Every non-empty recorded value comes back identical; returns values checked."""
    by_key = {(r["symbol"], r["recorded_at"]): r for r in back}
    assert len(by_key) == len(rows), f"{len(rows)} runs in, {len(by_key)} out"
    checked = 0
    for row in rows:
        out = by_key[(row["symbol"], row["recorded_at"])]
        assert out["model_version"] == infer_model_version(row), row["recorded_at"]
        expected = {k: v for k, v in row.items() if k not in _NOT_FIELDS and v not in (None, "")}
        actual = {k: v for k, v in out.items() if k not in _NOT_FIELDS}
        assert actual == expected, f"{row['symbol']} {row['recorded_at']}: values changed"
        checked += len(expected)
    return checked


def migrate(csv_path: Path | None = None, store_dir: Path | None = None) -> dict:
    csv_path = Path(csv_path or signal_snapshots.SNAPSHOT_PATH)
    store_dir = Path(store_dir or snapshot_store.STORE_DIR)
    if snapshot_store.active(store_dir):
        raise SystemExit(f"Already migrated ({store_dir / snapshot_store.MARKER_NAME}).")
    if not csv_path.exists():
        raise SystemExit(f"No snapshot CSV at {csv_path}.")
    if any(store_dir.glob("part-*.parquet")):
        raise SystemExit(f"{store_dir} already holds part files but no migration marker; "
                         f"inspect it before migrating, so nothing is stored twice.")
    snapshot_store._require_pyarrow()  # fail before reading anything, with the fix
    before = _sha256(csv_path)
    rows = _read_csv(csv_path)
    long = snapshot_store.wide_to_long(rows, snapshot_store.LIVE_RUN, infer_model_version)
    checked = assert_round_trip(rows, snapshot_store.long_to_wide(long))
    part = snapshot_store.append(long, store_dir)
    try:
        stored = snapshot_store.load(store_dir)
        assert len(stored) == len(long), "row count changed on disk"
        assert_round_trip(rows, snapshot_store.long_to_wide(stored))
        assert _sha256(csv_path) == before, "the CSV was modified"
    except BaseException:
        part.unlink(missing_ok=True)  # roll back: no half-migrated store
        raise
    summary = {
        "migrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": str(csv_path), "csv_sha256": before, "runs": len(rows),
        "long_rows": len(long), "values_checked": checked, "part": part.name,
        "schema_version": snapshot_store.SCHEMA_VERSION,
    }
    (store_dir / snapshot_store.MARKER_NAME).write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        print(json.dumps(migrate(), indent=2))
    except snapshot_store.StoreUnavailable as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
