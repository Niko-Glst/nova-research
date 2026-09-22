"""Local web UI: type a ticker, watch the analysis run, read the report.

    python serve.py
    python serve.py --port 9000 --no-open

Binds to 127.0.0.1 only. This is deliberate and not merely a default: the
process holds API credentials from .env and will happily fetch anything asked
of it, so it must not be reachable from the network. There is no authentication
here because there is no remote access to authenticate.

Built on http.server from the standard library rather than Flask, in keeping
with the rest of the project: the whole server is one handler with four routes,
and a framework would be more dependency than code.

Progress is streamed over Server-Sent Events. A full run takes one to three
minutes, nearly all of it rate-limit sleeping, and a page that sits silent for
that long reads as a hang. SSE is a better fit than polling here because the
server knows when each stage finishes and the browser does not need to ask.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from agents import html_report, research_report

# Ticker charset. Real symbols are letters with an optional dot or hyphen
# (BRK.B, RDS-A); anything else is rejected before it reaches a provider.
_TICKER_MAX_LENGTH = 12

# Completed reports, kept in memory so the result page can be re-read without
# re-running the analysis. Keyed by job id.
_reports: dict[str, research_report.ResearchReport] = {}
_reports_lock = threading.Lock()


def _valid_ticker(raw: str) -> str | None:
    """Return a normalized ticker, or None when the input is not one."""
    candidate = (raw or "").strip().upper()
    if not candidate or len(candidate) > _TICKER_MAX_LENGTH:
        return None
    if not all(character.isalnum() or character in ".-^" for character in candidate):
        return None
    if not any(character.isalpha() for character in candidate):
        return None
    return candidate


INDEX_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nova-research</title>
<style>
:root {
  --bg: #f6f7f9; --card: #fff; --ink: #1a1d21; --muted: #6b7280;
  --line: #e3e6ea; --accent: #1f4fd8; --good: #1a7f4b; --bad: #b3261e;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  min-height: 100vh; }
.wrap { max-width: 640px; margin: 0 auto; padding: 56px 20px; }
.brand { font-size: 10px; font-weight: 700; letter-spacing: 2.2px;
  color: var(--muted); margin-bottom: 8px; }
h1 { font-size: 32px; margin: 0 0 6px; letter-spacing: -0.6px; }
.tagline { color: var(--muted); margin: 0 0 28px; }
.card { background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 22px; }
label { display: block; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.6px; color: var(--muted); margin-bottom: 6px; }
.row { display: flex; gap: 10px; }
input[type=text] { flex: 1; font-size: 20px; padding: 11px 14px;
  border: 1px solid var(--line); border-radius: 7px; text-transform: uppercase;
  font-weight: 600; letter-spacing: 1px; min-width: 0; }
input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
button { background: var(--accent); color: #fff; border: 0; border-radius: 7px;
  padding: 11px 22px; font-size: 15px; font-weight: 600; cursor: pointer; }
button:hover:not(:disabled) { background: #1740b8; }
button:disabled { background: #9aa0a6; cursor: not-allowed; }
.options { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 16px; }
.opt { display: flex; align-items: center; gap: 7px; font-size: 14px; }
.opt input { width: 16px; height: 16px; accent-color: var(--accent); }
.hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
.examples { margin-top: 18px; font-size: 13px; color: var(--muted); }
.examples button { background: none; color: var(--accent); border: 0;
  padding: 0 4px; font-size: 13px; font-weight: 600; text-decoration: underline; }
#progress { margin-top: 20px; display: none; }
#progress.active { display: block; }
.stage { display: flex; align-items: flex-start; gap: 10px; padding: 7px 0;
  border-bottom: 1px solid var(--line); font-size: 14px; }
.stage:last-child { border-bottom: 0; }
.mark { width: 16px; flex-shrink: 0; font-weight: 700; }
.stage.running .mark::after { content: "\\2022"; color: var(--accent); }
.stage.done .mark::after { content: "\\2713"; color: var(--good); }
.stage.failed .mark::after { content: "\\2717"; color: var(--bad); }
.stage-name { font-weight: 600; width: 104px; flex-shrink: 0; }
.stage-msg { color: var(--muted); }
.elapsed { font-size: 12px; color: var(--muted); margin-top: 10px; }
.error { background: #fbeae9; color: var(--bad); padding: 11px 14px;
  border-radius: 7px; margin-top: 16px; font-size: 14px; }
footer { margin-top: 34px; text-align: center; font-size: 12px; color: var(--muted); }
@media (max-width: 520px) {
  .row { flex-direction: column; }
  .stage-name { width: auto; }
}
</style></head>
<body><div class="wrap">
<div class="brand">NIKOLAY GELSHTEIN</div>
<h1>nova-research</h1>
<p class="tagline">Four analysts, a falsifiable thesis, and a distribution of
outcomes. Research only &mdash; not advice.</p>

<div class="card">
  <form id="form">
    <label for="ticker">Ticker</label>
    <div class="row">
      <input type="text" id="ticker" name="ticker" placeholder="NVDA"
             autocomplete="off" autofocus maxlength="12" required>
      <button type="submit" id="go">Analyze</button>
    </div>
    <div class="options">
      <label class="opt"><input type="checkbox" id="peers" checked> Peer comparison</label>
      <label class="opt"><input type="checkbox" id="simulate" checked> Backtest &amp; probabilities</label>
    </div>
    <p class="hint">A full run takes one to three minutes, almost all of it
       waiting on provider rate limits. Unchecking both is roughly 15 seconds.</p>
  </form>

  <div class="examples">
    Try:
    <button type="button" class="example">NVDA</button>
    <button type="button" class="example">AAPL</button>
    <button type="button" class="example">GRND</button>
    <button type="button" class="example">MP</button>
  </div>

  <div id="progress">
    <div id="stages"></div>
    <div class="elapsed" id="elapsed"></div>
  </div>
  <div id="error" style="display:none" class="error"></div>
</div>

<footer>Runs locally. Reports are written to <code>reports/</code>.</footer>
</div>

<script>
const STAGES = [
  ["technical",   "Technical"],
  ["fundamental", "Fundamental"],
  ["sentiment",   "Sentiment"],
  ["insider",     "Insider"],
  ["simulation",  "Simulation"],
];

const form = document.getElementById("form");
const input = document.getElementById("ticker");
const button = document.getElementById("go");
const progress = document.getElementById("progress");
const stagesEl = document.getElementById("stages");
const elapsedEl = document.getElementById("elapsed");
const errorEl = document.getElementById("error");
let timer = null;

document.querySelectorAll(".example").forEach(b => {
  b.addEventListener("click", () => { input.value = b.textContent; form.requestSubmit(); });
});

function resetStages(includeSimulation) {
  stagesEl.innerHTML = "";
  STAGES.forEach(([key, label]) => {
    if (key === "simulation" && !includeSimulation) return;
    const row = document.createElement("div");
    row.className = "stage";
    row.id = "stage-" + key;
    row.innerHTML = '<span class="mark"></span>' +
      '<span class="stage-name">' + label + '</span>' +
      '<span class="stage-msg">waiting</span>';
    stagesEl.appendChild(row);
  });
}

function setStage(key, message, state) {
  const row = document.getElementById("stage-" + key);
  if (!row) return;
  row.className = "stage " + state;
  row.querySelector(".stage-msg").textContent = message;
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const ticker = input.value.trim().toUpperCase();
  if (!ticker) return;

  const peers = document.getElementById("peers").checked;
  const simulate = document.getElementById("simulate").checked;

  button.disabled = true;
  button.textContent = "Running";
  errorEl.style.display = "none";
  progress.classList.add("active");
  resetStages(simulate);

  const started = Date.now();
  clearInterval(timer);
  timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    elapsedEl.textContent = seconds + "s elapsed";
  }, 1000);

  const params = new URLSearchParams({
    ticker: ticker,
    peers: peers ? "1" : "0",
    simulate: simulate ? "1" : "0",
  });
  const source = new EventSource("/analyze?" + params.toString());

  source.addEventListener("progress", e => {
    const data = JSON.parse(e.data);
    const failed = /unavailable/i.test(data.message);
    setStage(data.stage, data.message, failed ? "failed" : "running");
    // Mark every earlier stage done: the pipeline runs them in order.
    const index = STAGES.findIndex(s => s[0] === data.stage);
    STAGES.slice(0, index).forEach(([key]) => {
      const row = document.getElementById("stage-" + key);
      if (row && row.classList.contains("running")) {
        row.className = "stage done";
      }
    });
  });

  source.addEventListener("done", e => {
    const data = JSON.parse(e.data);
    clearInterval(timer);
    source.close();
    window.location = data.url;
  });

  source.addEventListener("failed", e => {
    const data = JSON.parse(e.data);
    clearInterval(timer);
    source.close();
    button.disabled = false;
    button.textContent = "Analyze";
    errorEl.textContent = data.message;
    errorEl.style.display = "block";
    progress.classList.remove("active");
  });

  source.onerror = () => {
    clearInterval(timer);
    source.close();
    button.disabled = false;
    button.textContent = "Analyze";
    errorEl.textContent = "Lost connection to the local server.";
    errorEl.style.display = "block";
  };
});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    """Four routes: the form, the SSE analysis stream, a report, and a 404."""

    # Quieter logs: one line per request is enough for a local tool.
    def log_message(self, format: str, *args) -> None:
        print(f"  {self.address_string()} {format % args}")

    def do_GET(self) -> None:  # noqa: N802  (http.server's naming)
        route = urlparse(self.path)
        if route.path == "/":
            self._send_html(INDEX_PAGE)
        elif route.path == "/analyze":
            self._stream_analysis(parse_qs(route.query))
        elif route.path.startswith("/report/"):
            self._send_report(route.path.rsplit("/", 1)[-1])
        else:
            self._send_html("<h1>404</h1><p><a href='/'>Back</a></p>", status=404)

    # -- routes ------------------------------------------------------------

    def _stream_analysis(self, params: dict[str, list[str]]) -> None:
        """Run the analysis, streaming progress as Server-Sent Events."""
        ticker = _valid_ticker((params.get("ticker") or [""])[0])
        if ticker is None:
            self._send_sse_headers()
            self._emit("failed", {"message": "That does not look like a ticker symbol."})
            return

        include_peers = (params.get("peers") or ["1"])[0] == "1"
        simulate = (params.get("simulate") or ["1"])[0] == "1"

        self._send_sse_headers()

        # The analysis runs on a worker thread and pushes progress onto a queue,
        # so this thread can keep writing to the socket while it works. Doing
        # both on one thread would buffer every update until the end.
        events: queue.Queue = queue.Queue()

        def on_progress(stage: str, message: str) -> None:
            events.put(("progress", {"stage": stage, "message": message}))

        def worker() -> None:
            try:
                report = research_report.analyze(
                    ticker,
                    include_peers=include_peers,
                    run_simulation=simulate,
                    on_progress=on_progress,
                )
                with _reports_lock:
                    _reports[ticker] = report
                _write_report_file(report)
                events.put(("done", {"url": f"/report/{ticker}"}))
            except Exception as exc:
                traceback.print_exc()
                events.put(
                    ("failed", {"message": f"{type(exc).__name__}: {exc}"})
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                name, payload = events.get(timeout=30)
            except queue.Empty:
                # A comment line keeps the connection alive through a long
                # rate-limit sleep without the browser giving up on it.
                if not self._write(": keepalive\n\n"):
                    return
                continue

            if not self._emit(name, payload):
                return
            if name in ("done", "failed"):
                return

    def _send_report(self, ticker: str) -> None:
        with _reports_lock:
            report = _reports.get(ticker.upper())
        if report is None:
            self._send_html(
                "<h1>No report for that symbol yet</h1><p><a href='/'>Run one</a></p>",
                status=404,
            )
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        page = html_report.render_report(report, stamp)
        page = page.replace(
            "<body>",
            '<body><div style="max-width:900px;margin:0 auto;padding:14px 20px 0">'
            '<a href="/" style="font-size:13px;color:#6b7280">&larr; Analyze another'
            "</a></div>",
            1,
        )
        self._send_html(page)

    # -- plumbing ----------------------------------------------------------

    def _send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # Without this, a proxy in front would buffer the stream into silence.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _emit(self, name: str, payload: dict) -> bool:
        return self._write(f"event: {name}\ndata: {json.dumps(payload)}\n\n")

    def _write(self, text: str) -> bool:
        """Write to the socket, returning False once the client has gone."""
        try:
            self.wfile.write(text.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return False


def _write_report_file(report: research_report.ResearchReport) -> None:
    """Also save the report to reports/, matching what the CLI produces."""
    from pathlib import Path

    directory = Path("reports")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    (directory / f"{report.symbol}.html").write_text(
        html_report.render_report(report, stamp), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the research tool on localhost.",
        epilog="Research output only -- not financial advice.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a browser on start."
    )
    args = parser.parse_args(argv)

    # 127.0.0.1, never 0.0.0.0: this process holds API credentials and has no
    # authentication, so it must not be reachable from the network.
    address = ("127.0.0.1", args.port)
    server = ThreadingHTTPServer(address, Handler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"nova-research is serving at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
