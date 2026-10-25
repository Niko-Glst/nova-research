"""Atomic writes, and caches that survive a file cut short."""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from agents import sec_edgar
from backtests import market_data
from common.files import atomic_write, atomic_write_text


def test_a_failed_write_leaves_the_old_file_whole(tmp_path):
    path = atomic_write_text(tmp_path / "prices.csv", "date,close\n2026-10-23,100\n")
    original = path.read_bytes()

    def crash(temporary):
        temporary.write_text("date,cl")  # cut short
        raise OSError("disk full")

    with pytest.raises(OSError):
        atomic_write(path, crash)
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["prices.csv"]  # no temp left behind


def test_pandas_writes_through_the_helper(tmp_path):
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="date"))
    atomic_write(tmp_path / "x.csv", frame.to_csv)
    assert pd.read_csv(tmp_path / "x.csv", index_col="date")["close"].tolist() == [1.0, 2.0]


class TestSecCache:
    def test_a_truncated_cache_is_refetched_not_trusted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sec_edgar, "get_sec_user_agent", lambda: "test agent test@x.y")
        cache = tmp_path / "company_tickers.json"
        cache.write_text('{"0": {"cik_str": 32', encoding="utf-8")  # cut short, but fresh
        calls = []

        def fetcher(url, headers):
            calls.append(url)
            return json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL"}}).encode()

        data = sec_edgar.get_json(sec_edgar.TICKERS_URL, cache, 168, fetcher)
        assert calls and data["0"]["cik_str"] == 320193
        assert json.loads(cache.read_text(encoding="utf-8")) == data  # repaired

    def test_a_cache_write_failure_keeps_the_fetched_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sec_edgar, "get_sec_user_agent", lambda: "test agent test@x.y")

        def no_write(path, text):
            raise OSError("read-only")

        monkeypatch.setattr(sec_edgar, "atomic_write_text", no_write)
        data = sec_edgar.get_json(sec_edgar.TICKERS_URL, tmp_path / "t.json", 168,
                                  lambda url, headers: b'{"ok": 1}')
        assert data == {"ok": 1}

    def test_a_failed_fetch_with_a_corrupt_cache_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sec_edgar, "get_sec_user_agent", lambda: "test agent test@x.y")
        cache = tmp_path / "t.json"
        cache.write_text("{", encoding="utf-8")
        old = time.time() - 10 * 24 * 3600
        os.utime(cache, (old, old))

        def offline(url, headers):
            raise OSError("offline")

        with pytest.raises(OSError, match="offline"):
            sec_edgar.get_json(sec_edgar.TICKERS_URL, cache, 168, offline)


def test_ohlcv_cache_is_written_whole(tmp_path):
    days = pd.bdate_range("2026-10-01", periods=5, tz="America/New_York")
    history = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in market_data.OHLCV_COLUMNS},
                           index=days)
    frame = market_data.load_ohlcv("TEST", cache_dir=tmp_path,
                                   fetcher=lambda symbol, period: history)
    again = pd.read_csv(tmp_path / "TEST.csv", index_col="date", parse_dates=["date"])
    assert len(again) == len(frame) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["TEST.csv", "TEST.meta.json"]
