"""The research.py command: which runs reach the track record, and asking for peers."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

import research
from agents import fa_analyst, research_report


def _report(peer_verdict: str | None = "cheap") -> research_report.ResearchReport:
    peers = SimpleNamespace(verdict=peer_verdict) if peer_verdict else None
    report = research_report.ResearchReport(symbol="GOOD", verdict="positive", score=0.4,
                                            confidence=1.0)
    object.__setattr__(report, "fundamental", SimpleNamespace(peer_comparison=peers))
    return report


@pytest.fixture
def run(monkeypatch):
    """research.main with a canned report; yields what it recorded and passed on."""
    state = SimpleNamespace(report=_report(), recorded=[], options={})

    def analyze(symbol, **options):
        state.options = options
        return state.report

    monkeypatch.setattr(research.research_report, "analyze", analyze)
    monkeypatch.setattr(research.research_report, "format_report", lambda report: "text")
    monkeypatch.setattr(research.signal_snapshots, "record", state.recorded.append)
    return state


def _terminal(monkeypatch, typed: str = "") -> None:
    stdin = io.StringIO(typed)
    stdin.isatty = lambda: True
    monkeypatch.setattr(research.sys, "stdin", stdin)


class TestTrackRecord:
    def test_a_run_with_a_peer_read_is_recorded(self, run, capsys):
        assert research.main(["GOOD"]) == 0
        assert run.recorded == [run.report]
        assert "track record" not in capsys.readouterr().err

    @pytest.mark.parametrize("peer_verdict", [None, "unknown"])
    def test_a_run_without_one_stays_out_and_says_so(self, run, capsys, peer_verdict):
        run.report = _report(peer_verdict)
        assert research.main(["GOOD"]) == 0
        assert run.recorded == []
        assert "not added to the track record" in capsys.readouterr().err

    def test_a_failed_fundamental_read_stays_out(self, run):
        object.__setattr__(run.report, "fundamental", None)
        assert research.main(["GOOD"]) == 0
        assert run.recorded == []


class TestWhenItAsks:
    def test_at_a_terminal(self, run, monkeypatch):
        _terminal(monkeypatch)
        research.main(["GOOD"])
        assert run.options["ask_peers"] is research._ask_peers

    def test_not_when_turned_off(self, run, monkeypatch):
        _terminal(monkeypatch)
        research.main(["GOOD", "--no-ask-peers"])
        assert run.options["ask_peers"] is None

    def test_not_without_a_terminal(self, run, monkeypatch):
        stdin = io.StringIO("")
        stdin.isatty = lambda: False
        monkeypatch.setattr(research.sys, "stdin", stdin)
        research.main(["GOOD"])
        assert run.options["ask_peers"] is None


class TestAskPeers:
    @pytest.fixture(autouse=True)
    def offline(self, tmp_path, monkeypatch):
        """Yahoo knows every ticker but NOPE; choices go to a temporary file."""
        def fetch(ticker):
            if ticker == "NOPE":
                raise ValueError("no data")
            return {}

        monkeypatch.setattr(fa_analyst, "fetch_fundamentals", fetch)
        monkeypatch.setattr(fa_analyst, "CUSTOM_PEERS_PATH", tmp_path / "custom_peers.yaml")

    def test_commas_spaces_and_case_are_all_fine_and_the_answer_is_saved(self, monkeypatch):
        _terminal(monkeypatch, "aaa, bbb.as ccc aaa\n")
        assert research._ask_peers("GOOD", "Healthcare", ["JNJ"]) == ["AAA", "BBB.AS", "CCC"]
        assert fa_analyst.load_custom_peers() == {"GOOD": ["AAA", "BBB.AS", "CCC"]}

    @pytest.mark.parametrize("typed", ["\n", ""])  # Enter, and the end of the input
    def test_enter_keeps_what_there_is_and_saves_nothing(self, monkeypatch, typed):
        _terminal(monkeypatch, typed)
        assert research._ask_peers("GOOD", "Healthcare", ["JNJ"]) is None
        assert fa_analyst.load_custom_peers() == {}

    def test_an_unknown_ticker_is_named_and_asked_again(self, monkeypatch, capsys):
        _terminal(monkeypatch, "AAA NOPE\nAAA\n")
        assert research._ask_peers("GOOD", "Healthcare", []) == ["AAA"]
        assert "No data for NOPE" in capsys.readouterr().err

    def test_something_that_is_no_ticker_never_reaches_yahoo(self, monkeypatch, capsys):
        looked_up = []
        monkeypatch.setattr(fa_analyst, "fetch_fundamentals", looked_up.append)
        _terminal(monkeypatch, "../etc/passwd\n\n")
        assert research._ask_peers("GOOD", "Healthcare", []) is None
        assert looked_up == []
        assert "No data for ../ETC/PASSWD" in capsys.readouterr().err

    def test_a_company_cannot_be_its_own_peer(self, monkeypatch, capsys):
        _terminal(monkeypatch, "good\nAAA\n")
        assert research._ask_peers("GOOD", "Healthcare", []) == ["AAA"]
        assert "its own peer" in capsys.readouterr().err

    def test_too_many_are_cut_to_the_limit_with_a_message(self, monkeypatch, capsys):
        monkeypatch.setattr(fa_analyst, "max_peers", lambda: 2)
        _terminal(monkeypatch, "AAA BBB CCC\n")
        assert research._ask_peers("GOOD", "Healthcare", []) == ["AAA", "BBB"]
        assert "first 2" in capsys.readouterr().err

    def test_a_failed_save_still_uses_the_answer(self, monkeypatch, capsys):
        def locked(symbol, peers):
            raise PermissionError("in use")

        monkeypatch.setattr(fa_analyst, "save_custom_peers", locked)
        _terminal(monkeypatch, "AAA\n")
        assert research._ask_peers("GOOD", "Healthcare", []) == ["AAA"]
        assert "not saved" in capsys.readouterr().err

    def test_the_question_never_mixes_into_json_output(self, monkeypatch, capsys):
        _terminal(monkeypatch, "AAA\n")
        research._ask_peers("GOOD", "Healthcare", [])
        assert capsys.readouterr().out == ""

    def test_json_mode_stays_parseable_when_it_skips_the_record(self, run, capsys):
        run.report = _report(None)
        object.__setattr__(run.report, "fundamental", None)
        assert research.main(["GOOD", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)[0]["symbol"] == "GOOD"
