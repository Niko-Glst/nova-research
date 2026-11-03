"""Every test gets its own empty signal history, never the real one."""

from __future__ import annotations

import pytest

from agents import signal_snapshots, snapshot_store


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Once the store is migrated, every recorded run goes to it, whatever path a
    caller passes, so without this a test would add its rows to the track record."""
    monkeypatch.setattr(snapshot_store, "STORE_DIR", tmp_path / "snapshot_store")
    monkeypatch.setattr(signal_snapshots, "SNAPSHOT_PATH", tmp_path / "signals.csv")
    monkeypatch.setattr(signal_snapshots, "RESCORED_PATH", tmp_path / "rescored.csv")
