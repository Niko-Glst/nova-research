"""A throttled, empty answer from Yahoo must not cost an analysis its data."""

from __future__ import annotations

import pandas as pd
import pytest

from agents import ta_analyst


class _Ticker:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def history(self, period, auto_adjust):
        self.calls += 1
        return self.answers.pop(0)


@pytest.fixture
def no_wait(monkeypatch):
    class _Limiter:
        def wait(self):
            pass

    waits = []
    monkeypatch.setattr(ta_analyst, "get_limiter", lambda name: _Limiter())
    monkeypatch.setattr(ta_analyst, "_sleep", waits.append)
    monkeypatch.setattr(ta_analyst, "_known_empty", set())
    return waits


GOOD = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.bdate_range("2026-10-01", periods=2))


def test_empty_answers_are_retried_with_growing_waits(monkeypatch, no_wait):
    """NBIS, 2026-10-30: two empty answers mid-run, then 509 rows."""
    fake = _Ticker([pd.DataFrame(), pd.DataFrame(), GOOD])
    monkeypatch.setattr(ta_analyst.yf, "Ticker", lambda symbol: fake)
    assert ta_analyst.fetch_price_history("NBIS", "5y")["Close"].tolist() == [1.0, 2.0]
    assert fake.calls == 3 and no_wait == list(ta_analyst.EMPTY_HISTORY_WAITS)


def test_a_symbol_that_stays_empty_fails_and_is_remembered(monkeypatch, no_wait):
    attempts = len(ta_analyst.EMPTY_HISTORY_WAITS) + 1
    fake = _Ticker([pd.DataFrame()] * (attempts + 1))
    monkeypatch.setattr(ta_analyst.yf, "Ticker", lambda symbol: fake)
    with pytest.raises(ValueError, match="No price history"):
        ta_analyst.fetch_price_history("ZZZZZ")
    assert fake.calls == attempts
    with pytest.raises(ValueError):  # the second ask fails at once, no waiting again
        ta_analyst.fetch_price_history("ZZZZZ")
    assert fake.calls == attempts + 1
    assert len(no_wait) == len(ta_analyst.EMPTY_HISTORY_WAITS)
