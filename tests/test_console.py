"""Redirected output must not crash on characters the console encoding lacks."""

from __future__ import annotations

import io

import pytest

from common.console import safe_console


def test_cp1252_stream_prints_emoji_as_replacement():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("moon \U0001F680")
        stream.flush()
    raw.seek(0), raw.truncate()
    safe_console(stream)
    stream.write("moon \U0001F680 ±1.2%")
    stream.flush()
    assert raw.getvalue() == "moon ? ±1.2%".encode("cp1252")


def test_streams_without_reconfigure_are_left_alone():
    safe_console(io.StringIO(), object())
