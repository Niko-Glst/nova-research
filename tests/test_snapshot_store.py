"""Snapshot hardening: NaN not 0, the long format, the parquet store and the migration."""

from __future__ import annotations

import csv
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agents import signal_snapshots, snapshot_store

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "migrate_snapshots_to_long", ROOT / "scripts" / "migrate_snapshots_to_long.py")
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)

OLD_COLUMNS = signal_snapshots.COLUMNS[:25]
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
needs_pyarrow = pytest.mark.skipif(not snapshot_store.pyarrow_status()[0],
                                   reason="pyarrow not installed (see snapshot_store.PYARROW_FIX)")


def _legacy_csv(path: Path) -> list[list[str]]:
    rows = [
        ["2026-10-15T16:37:11+00:00", "BESI.AS", "negative", "-0.4444444444444445",
         "0.6428571428571429", "bearish", "fair", "unknown", "unknown", "163.39999389648438"]
        + [""] * 15,
        ["2026-10-15T16:38:18+00:00", "BESI.AS", "negative", "-0.4444444444444445",
         "0.6428571428571429", "bearish", "fair", "unknown", "unknown", "163.39999389648438"]
        + [""] * 15,
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OLD_COLUMNS)
        writer.writerows(rows)
    return rows


def _report():
    return SimpleNamespace(symbol="IBRX", verdict="negative", score=-0.375, confidence=1.0,
                           technical=None, fundamental=None, sentiment=None, insider=None,
                           flows=None, model_version="2.0.0")


class TestTheRealHistoryIsOutOfReach:
    def test_every_default_location_is_temporary(self, tmp_path):
        for path in (snapshot_store.STORE_DIR, signal_snapshots.SNAPSHOT_PATH,
                     signal_snapshots.RESCORED_PATH):
            assert path.is_relative_to(tmp_path)

    @needs_pyarrow
    def test_a_recorded_run_lands_there_even_in_an_active_store(self, tmp_path):
        """The bug this guards against: a test wrote TEST rows into the migrated store."""
        store = snapshot_store.STORE_DIR
        store.mkdir()
        (store / snapshot_store.MARKER_NAME).write_text("{}", encoding="utf-8")
        assert snapshot_store.active()
        signal_snapshots.record(_report(), tmp_path / "ignored.csv", NOW)
        assert len(list(store.glob("part-*.parquet"))) == 1
        assert [r["symbol"] for r in signal_snapshots.load_rows()] == ["IBRX"]
        assert not snapshot_store.load().empty


class TestNanNotZero:
    def test_widened_columns_load_as_nan(self, tmp_path):
        path = tmp_path / "signals.csv"
        _legacy_csv(path)
        signal_snapshots.record(_report(), path, NOW, store_dir=tmp_path / "long")
        frame = pd.read_csv(path)
        assert list(frame.columns) == list(signal_snapshots.COLUMNS)
        old = frame.iloc[:2]
        for column in ("net_issuance", "stock_vol_z", "market_vol_ratio", "atm_implied_vol"):
            assert old[column].isna().all(), column  # NaN, never 0
        assert (frame["schema_version"].iloc[2] == 1) and pd.isna(frame["schema_version"].iloc[0])

    def test_missing_long_fields_pivot_to_nan(self):
        rows = [{"recorded_at": "a", "symbol": "X", "score": "0.1", "net_issuance": "0.02"},
                {"recorded_at": "b", "symbol": "Y", "score": "-0.3"}]
        long = snapshot_store.wide_to_long(rows, "live_run", lambda r: "2.0.0")
        wide = long.pivot_table(index="ticker", columns="field", values="value", aggfunc="first")
        assert pd.isna(wide.loc["Y", "net_issuance"])


class TestLongFormat:
    def test_round_trip_keeps_every_value_and_drops_empties(self, tmp_path):
        path = tmp_path / "signals.csv"
        _legacy_csv(path)
        rows = migration._read_csv(path)
        long = snapshot_store.wide_to_long(rows, "live_run", migration.infer_model_version)
        assert set(long.columns) == set(snapshot_store.LONG_COLUMNS)
        assert (long["value"] != "").all() and (long["schema_version"] == 2).all()
        assert set(long["model_version"]) == {"1.0.0"}  # no flows leg: pre-1.1
        assert set(long["date"]) == {"2026-10-15"}
        # Two runs a minute apart stay apart.
        assert migration.assert_round_trip(rows, snapshot_store.long_to_wide(long)) == 2 * 8

    def test_a_changed_value_fails_the_check(self, tmp_path):
        path = tmp_path / "signals.csv"
        _legacy_csv(path)
        rows = migration._read_csv(path)
        back = snapshot_store.long_to_wide(
            snapshot_store.wide_to_long(rows, "live_run", migration.infer_model_version))
        back[0]["score"] = "-0.44"
        with pytest.raises(AssertionError, match="values changed"):
            migration.assert_round_trip(rows, back)

    def test_missing_pyarrow_is_reported_with_the_fix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(snapshot_store, "pyarrow_status", lambda: (False, "ImportError: x"))
        frame = snapshot_store.wide_to_long([{"recorded_at": "a", "symbol": "X", "s": "1"}],
                                            "live_run", lambda r: "2.0.0")
        with pytest.raises(snapshot_store.StoreUnavailable, match="pip install pyarrow"):
            snapshot_store.append(frame, tmp_path)


@needs_pyarrow
class TestParquetStore:
    def test_append_only_parts_and_exact_values(self, tmp_path):
        rows = [{"recorded_at": "2026-10-22T12:00:00+00:00", "symbol": "X",
                 "score": "-0.37499999999999994", "verdict": "negative"}]
        first = snapshot_store.append(
            snapshot_store.wide_to_long(rows, "live_run", lambda r: "2.0.0"), tmp_path)
        before = first.read_bytes()
        snapshot_store.append(snapshot_store.wide_to_long(
            [{**rows[0], "recorded_at": "2026-10-23T12:00:00+00:00"}], "live_run",
            lambda r: "2.0.0"), tmp_path)
        assert first.read_bytes() == before  # never rewritten
        back = snapshot_store.long_to_wide(snapshot_store.load(tmp_path))
        assert back[0]["score"] == "-0.37499999999999994"  # exact text, no float rounding
        assert len(back) == 2

    def test_migration_end_to_end(self, tmp_path):
        path = tmp_path / "signals.csv"
        _legacy_csv(path)
        original = path.read_bytes()
        summary = migration.migrate(path, tmp_path / "long")
        assert summary["runs"] == 2 and summary["values_checked"] == 16
        assert path.read_bytes() == original
        assert snapshot_store.active(tmp_path / "long")
        with pytest.raises(SystemExit, match="Already migrated"):
            migration.migrate(path, tmp_path / "long")
        # After migration, new runs go to the long store.
        signal_snapshots.record(_report(), path, NOW, store_dir=tmp_path / "long")
        assert path.read_bytes() == original
        rows = signal_snapshots.load_rows(path, store_dir=tmp_path / "long")
        assert [r["symbol"] for r in rows] == ["BESI.AS", "BESI.AS", "IBRX"]
