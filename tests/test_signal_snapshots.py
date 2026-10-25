"""Signal snapshots: flow columns, and widening an older file without changing a value."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from types import MappingProxyType, SimpleNamespace

import pytest

from agents import flow_analyst, signal_snapshots

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
OLD_COLUMNS = signal_snapshots.COLUMNS[:25]  # the header before the flow columns


def _report(flows=None):
    return SimpleNamespace(symbol="TEST", verdict="mixed", score=0.1, confidence=0.9,
                           technical=None, fundamental=None, sentiment=None, insider=None,
                           flows=flows)


def _flows():
    parts = (
        flow_analyst.FlowComponent("issuance", "Net issuance", -1, "",
                                   MappingProxyType({"net_issuance": 0.171}), "confirmed_zero"),
        flow_analyst.FlowComponent("systematic_market", "Market", 1, "",
                                   MappingProxyType({"up_horizons": 3.0, "vol_ratio": 0.72})),
        flow_analyst.FlowComponent("systematic_stock", "Stock", 1, "",
                                   MappingProxyType({"vol_ratio": 0.79})),
    )
    return flow_analyst.FlowRead(symbol="TEST", signal="neutral", score=1 / 3, components=parts,
                                 institutions_pct=0.2)


def _read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_flow_columns_are_filled():
    row = signal_snapshots.snapshot_row(_report(_flows()), NOW)
    assert row["flows_signal"] == "neutral"
    assert float(row["net_issuance"]) == pytest.approx(0.171)
    assert row["buyback_flag"] == "confirmed_zero"
    assert row["net_buyback_yield"] == ""  # retired with model 2.0.0
    assert float(row["market_vol_ratio"]) == pytest.approx(0.72)
    assert row["put_call_oi_ratio"] == ""  # no options: empty, not zero


def test_report_without_flows_still_records():
    row = signal_snapshots.snapshot_row(SimpleNamespace(
        symbol="X", verdict="mixed", score=0.0, confidence=1.0, technical=None,
        fundamental=None, sentiment=None, insider=None), NOW)
    assert row["flows_signal"] == ""


def test_an_older_file_is_widened_and_no_old_value_changes(tmp_path):
    path = tmp_path / "signals.csv"
    old_row = ["2026-10-15T18:37:00+00:00", "BESI.AS", "mixed", "0.1", "1.0"] + ["7"] * 20
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OLD_COLUMNS)
        writer.writerow(old_row)

    signal_snapshots.record(_report(_flows()), path, NOW)
    rows = _read(path)
    assert rows[0] == list(signal_snapshots.COLUMNS)
    assert rows[1][: len(old_row)] == old_row  # every old value unchanged
    assert rows[1][len(old_row):] == [""] * (len(signal_snapshots.COLUMNS) - len(old_row))
    assert rows[2][signal_snapshots.COLUMNS.index("flows_signal")] == "neutral"
    assert not path.with_suffix(".widening").exists()


def test_an_unrecognised_header_is_never_appended_to(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text("something,else\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not recognise"):
        signal_snapshots.record(_report(), path, NOW)
    assert path.read_text(encoding="utf-8") == "something,else\n1,2\n"
