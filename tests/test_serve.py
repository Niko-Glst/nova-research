"""Tests for the local web UI.

The server is exercised through real HTTP against a socket on an ephemeral
port, not by calling handler methods directly: routing, status codes and the
SSE framing are the things worth testing, and all three live in the parts a
direct call would skip.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import serve


@pytest.fixture
def server():
    """A server on an ephemeral port, torn down after the test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


class TestTickerValidation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("nvda", "NVDA"),
            ("  aapl  ", "AAPL"),
            ("brk.b", "BRK.B"),
            ("RDS-A", "RDS-A"),
        ],
    )
    def test_valid_tickers_are_normalized(self, raw, expected):
        assert serve._valid_ticker(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "123",  # no letters
            "A" * 20,  # too long
            "<script>alert(1)</script>",
            "../../etc/passwd",
            "NVDA;rm -rf /",
            "NVDA AAPL",  # space
        ],
    )
    def test_invalid_input_is_rejected(self, raw):
        """Anything that is not a symbol must not reach a data provider."""
        assert serve._valid_ticker(raw) is None

    def test_none_is_handled(self):
        assert serve._valid_ticker(None) is None


class TestRoutes:
    def test_index_serves_the_form(self, server):
        status, body = _get(server + "/")
        assert status == 200
        assert 'id="ticker"' in body
        assert "nova-research" in body

    def test_index_carries_the_branding(self, server):
        _, body = _get(server + "/")
        assert "NIKOLAY GELSHTEIN" in body

    def test_index_states_it_is_not_advice(self, server):
        _, body = _get(server + "/")
        assert "not advice" in body.lower()

    def test_unknown_route_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _get(server + "/nope")
        assert caught.value.code == 404

    def test_report_for_unknown_symbol_is_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _get(server + "/report/ZZZZ")
        assert caught.value.code == 404

    def test_report_renders_a_stored_result(self, server):
        from agents.research_report import ResearchReport

        report = ResearchReport(
            symbol="TEST", verdict="mixed", score=0.0, confidence=0.5
        )
        with serve._reports_lock:
            serve._reports["TEST"] = report
        try:
            status, body = _get(server + "/report/TEST")
            assert status == 200
            assert "TEST" in body
            assert "Analyze another" in body
        finally:
            with serve._reports_lock:
                serve._reports.pop("TEST", None)


class TestAnalyseStream:
    def test_invalid_ticker_emits_a_failure_event(self, server):
        status, body = _get(server + "/analyze?ticker=%3Cscript%3E")
        assert status == 200  # SSE always opens; the error is in the stream
        assert "event: failed" in body
        payload = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
        assert "ticker" in payload["message"].lower()

    def test_missing_ticker_emits_a_failure_event(self, server):
        _, body = _get(server + "/analyze")
        assert "event: failed" in body

    def test_stream_sets_event_stream_content_type(self, server):
        with urllib.request.urlopen(
            server + "/analyze?ticker=notaticker123456", timeout=10
        ) as response:
            assert "text/event-stream" in response.headers["Content-Type"]


class TestBindingIsLocalOnly:
    def test_main_binds_to_loopback(self, monkeypatch):
        """The process holds API credentials and has no auth, so it must not
        be reachable from the network.

        Asserted by capturing the address main() actually binds, rather than by
        grepping the source: a comment explaining why 0.0.0.0 is avoided would
        fail a textual check while the behaviour is correct.
        """
        bound: list[tuple] = []

        class FakeServer:
            def __init__(self, address, handler):
                bound.append(address)
                self.server_address = address

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                pass

        monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeServer)
        serve.main(["--port", "0", "--no-open"])

        assert bound == [("127.0.0.1", 0)]
