"""Failure paths: missing pyarrow, half-finished migrations, file errors in the web server.

Each test simulates the failure, so none needs pyarrow or the network.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agents import snapshot_store

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "migrate_snapshots_to_long", ROOT / "scripts" / "migrate_snapshots_to_long.py")
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


def _csv(path: Path) -> bytes:
    path.write_text("recorded_at,symbol,verdict,score\n"
                    "2026-10-15T16:37:11+00:00,BESI.AS,negative,-0.44\n", encoding="utf-8")
    return path.read_bytes()


class TestMigrationGuards:
    def test_without_pyarrow_nothing_is_read_or_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(snapshot_store, "pyarrow_status", lambda: (False, "ImportError: x"))
        original = _csv(tmp_path / "signals.csv")
        with pytest.raises(snapshot_store.StoreUnavailable, match="pip install pyarrow"):
            migration.migrate(tmp_path / "signals.csv", tmp_path / "long")
        assert (tmp_path / "signals.csv").read_bytes() == original
        assert not (tmp_path / "long").exists()

    def test_stray_part_files_stop_a_second_copy(self, tmp_path):
        _csv(tmp_path / "signals.csv")
        (tmp_path / "long").mkdir()
        (tmp_path / "long" / "part-x.parquet").write_bytes(b"")
        with pytest.raises(SystemExit, match="no migration marker"):
            migration.migrate(tmp_path / "signals.csv", tmp_path / "long")

    def test_a_failed_check_rolls_the_part_back(self, tmp_path, monkeypatch):
        _csv(tmp_path / "signals.csv")
        store = tmp_path / "long"
        monkeypatch.setattr(snapshot_store, "pyarrow_status", lambda: (True, ""))

        def fake_append(frame, store_dir):
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            part = Path(store_dir) / "part-1.parquet"
            part.write_bytes(b"written")
            return part

        monkeypatch.setattr(snapshot_store, "append", fake_append)
        monkeypatch.setattr(snapshot_store, "load",  # the read-back loses the rows
                            lambda store_dir, source=None: pd.DataFrame(
                                columns=list(snapshot_store.LONG_COLUMNS)))
        with pytest.raises(AssertionError, match="row count changed"):
            migration.migrate(tmp_path / "signals.csv", store)
        assert not list(store.glob("part-*.parquet"))
        assert not snapshot_store.active(store)


class TestHistoryWithoutPyarrow:
    def test_backfill_stops_before_any_download(self, monkeypatch, capsys):
        import history

        monkeypatch.setattr(snapshot_store, "pyarrow_status", lambda: (False, "ImportError: x"))

        def no_download(symbols):
            raise AssertionError("downloaded before checking the store")

        monkeypatch.setattr(history, "compute_backfill", no_download)
        assert history.main(["backfill"]) == 1
        assert "pip install pyarrow" in capsys.readouterr().err

    def test_store_backtest_points_to_from_sources(self, monkeypatch, capsys):
        import history

        def unavailable(*args, **kwargs):
            raise snapshot_store.StoreUnavailable("needs pyarrow")

        monkeypatch.setattr(snapshot_store, "load", unavailable)
        assert history.main(["backtest"]) == 1
        assert "--from-sources" in capsys.readouterr().err


class TestServerFileWrites:
    def test_write_failures_are_reported_not_raised(self, monkeypatch, capsys):
        import serve

        def broken(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(serve.report_archive, "save_report", broken)
        monkeypatch.setattr(serve.signal_snapshots, "record", broken)
        serve._write_report_file(SimpleNamespace(symbol="TEST"))
        err = capsys.readouterr().err
        assert "report file not saved" in err and "signal snapshot not saved" in err

    def test_writes_take_turns_and_use_the_project_reports_folder(self, monkeypatch):
        import serve

        active, overlap, seen = [0], [False], []

        def slow_save(report, directory):
            seen.append(Path(directory))
            active[0] += 1
            overlap[0] |= active[0] > 1
            threading.Event().wait(0.05)
            active[0] -= 1

        monkeypatch.setattr(serve.report_archive, "save_report", slow_save)
        monkeypatch.setattr(serve.signal_snapshots, "record", lambda report: None)
        threads = [threading.Thread(target=serve._write_report_file,
                                    args=(SimpleNamespace(symbol=f"T{i}"),)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not overlap[0]
        assert set(seen) == {ROOT / "reports"}
