"""Keep every HTML report, and an index page that links them all.

Each run writes three things into the reports directory:

- `<SYMBOL>.html`              the latest report, overwritten each run
- `archive/<SYMBOL>_<stamp>.html`  a dated copy that is never overwritten
- `index.html`                 every archived run, grouped by ticker

The index is rebuilt from the archived files themselves rather than from a
separate ledger, so it cannot drift out of sync with what is on disk: delete a
file and it drops off the index on the next run.

Research output only -- nothing here places an order.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from agents import html_report, signal_snapshots
from agents.research_report import MODEL_VERSION as CURRENT_MODEL
from common.files import atomic_write_text
from config.settings import load_scoring_config

ARCHIVE_DIR = "archive"

# A rescore is matched to its archived report within this many minutes.
RESCORE_MATCH_MINUTES = int(load_scoring_config()["history"]["rescore_match_minutes"])

# Model versions inferred for reports written before versioning (2026-10-22).
INFERRED_BASE = "1.0.0"
INFERRED_FLOWS = "1.1.0"
INDEX_FILE = "index.html"
_STAMP_FORMAT = "%Y-%m-%d_%H%M"
_FILE_PATTERN = re.compile(r"^(?P<symbol>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{4})\.html$")


@dataclass(frozen=True)
class ArchivedRun:
    """One archived report, with the headline figures read back out of it."""

    symbol: str
    generated: datetime
    path: Path
    verdict: str
    score: float | None
    confidence: int | None
    model_version: str = ""  # "" when the report predates versioning
    rescored: float | None = None  # score under the current model, when rescorable


def save_report(report, directory: str | Path = "reports", now: datetime | None = None) -> Path:
    """Write the latest copy and a dated archive copy, then rebuild the index.

    Returns the path of the latest copy (`<SYMBOL>.html`).
    """
    now = now or datetime.now()
    out_dir = Path(directory)
    archive_dir = out_dir / ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)

    document = html_report.render_report(report, now.strftime("%Y-%m-%d %H:%M"))
    latest = out_dir / f"{report.symbol}.html"
    # Atomic: the index reads verdicts back from these pages.
    atomic_write_text(latest, document)
    atomic_write_text(archive_dir / f"{report.symbol}_{now.strftime(_STAMP_FORMAT)}.html",
                      document)

    write_index(out_dir)
    return latest


def _read_run(path: Path) -> ArchivedRun | None:
    """Parse an archived report's filename and headline figures."""
    match = _FILE_PATTERN.match(path.name)
    if not match:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")

    verdict = re.search(r'<span class="verdict">([^<]+)</span>', text)
    score = re.search(r"score ([+-]?\d+\.\d+)", text)
    confidence = re.search(r"confidence (\d+)%", text)
    version = re.search(r'class="pill muted model-version">model ([^<]+)<', text)
    if version:
        model_version = version[1].strip()
    else:  # before versioning: the flows section arrived with 1.1.0
        model_version = INFERRED_FLOWS if "Market flows" in text else INFERRED_BASE
    return ArchivedRun(
        symbol=match["symbol"],
        generated=datetime.strptime(match["stamp"], _STAMP_FORMAT),
        path=path,
        verdict=verdict[1].strip() if verdict else "?",
        score=float(score[1]) if score else None,
        confidence=int(confidence[1]) if confidence else None,
        model_version=model_version,
    )


def _rescored_score(run: ArchivedRun, rescores: list[dict[str, str]]) -> float | None:
    """The current-model score of a run, matched to its snapshot by symbol and time.

    Archive stamps are local time and snapshots UTC, so both are compared in
    local time, within RESCORE_MATCH_MINUTES.
    """
    for r in rescores:
        if r.get("symbol") != run.symbol or r.get("model_version") != CURRENT_MODEL:
            continue
        try:
            recorded = datetime.fromisoformat(r["recorded_at"]).astimezone().replace(tzinfo=None)
            score = float(r["score"])
        except (KeyError, ValueError):
            continue
        if abs((recorded - run.generated).total_seconds()) <= RESCORE_MATCH_MINUTES * 60:
            return score
    return None


def list_runs(directory: str | Path = "reports") -> dict[str, list[ArchivedRun]]:
    """All archived runs, grouped by symbol, newest run first within each."""
    archive_dir = Path(directory) / ARCHIVE_DIR
    runs: dict[str, list[ArchivedRun]] = {}
    rescores = signal_snapshots.load_rescores()
    for path in sorted(archive_dir.glob("*.html")) if archive_dir.exists() else []:
        run = _read_run(path)
        if run is not None:
            if run.model_version != CURRENT_MODEL:
                run = replace(run, rescored=_rescored_score(run, rescores))
            runs.setdefault(run.symbol, []).append(run)
    for symbol_runs in runs.values():
        symbol_runs.sort(key=lambda r: r.generated, reverse=True)
    return dict(sorted(runs.items()))


def _verdict_tone(verdict: str) -> str:
    return html_report._tone(html_report._VERDICT_TONE, verdict.lower())


def _row(run: ArchivedRun, previous: ArchivedRun | None, root: Path) -> str:
    href = html.escape(run.path.relative_to(root).as_posix())
    score = f"{run.score:+.2f}" if run.score is not None else "&ndash;"
    confidence = f"{run.confidence}%" if run.confidence is not None else "&ndash;"

    change = "&ndash;"
    change_tone = ""
    comparable = previous is not None and previous.model_version == run.model_version
    if comparable and run.score is not None and previous.score is not None:
        delta = run.score - previous.score
        change = f"{delta:+.2f}"
        change_tone = "good" if delta > 0 else "bad" if delta < 0 else ""

    row = (
        f'<tr><td><a href="{href}">{run.generated:%Y-%m-%d %H:%M}</a></td>'
        f'<td><span class="pill {_verdict_tone(run.verdict)}">{html.escape(run.verdict)}</span></td>'
        f'<td class="num">{score}</td>'
        f'<td class="num {change_tone}">{change}</td>'
        f'<td class="num">{confidence}</td>'
        f'<td class="num">{html.escape(run.model_version)}</td></tr>'
    )
    if previous is not None and previous.model_version != run.model_version:
        # A visible break: scores either side are on different models and are not
        # compared or connected.
        row += (
            f'<tr class="version-break"><td colspan="6"><span class="pill warn">version break'
            f'</span> model {html.escape(previous.model_version)} &rarr; '
            f'{html.escape(run.model_version)}: scores before and after are not comparable'
            + (f"; under the current model the earlier run scores {previous.rescored:+.2f}"
               if previous.rescored is not None else
               "; the earlier run lacks the inputs to rescore it") + "</td></tr>"
        )
    return row


def render_index(directory: str | Path = "reports") -> str:
    root = Path(directory)
    runs = list_runs(root)

    sections = []
    for symbol, symbol_runs in runs.items():
        # Each run is compared with the one before it (the next entry, since
        # the list is newest first).
        rows = "".join(
            _row(run, symbol_runs[i + 1] if i + 1 < len(symbol_runs) else None, root)
            for i, run in enumerate(symbol_runs)
        )
        count = len(symbol_runs)
        sections.append(
            f"""
    <section class="card">
      <h2>{html.escape(symbol)} <span class="pill muted">{count} run{"s" if count != 1 else ""}</span></h2>
      <table>
        <thead><tr><th>Generated</th><th>Verdict</th><th class="num">Score</th>
        <th class="num">Change</th><th class="num">Confidence</th>
        <th class="num">Model</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""
        )

    body = "".join(sections) or (
        '<section class="card"><p class="muted-text">No reports yet. Run '
        "<code>python research.py TICKER --html</code> to create one.</p></section>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research reports &middot; Nikolay Gelshtein</title>
<style>{html_report._STYLE}</style></head>
<body><div class="wrap">
<header>
  <div class="brand">NIKOLAY GELSHTEIN</div>
  <h1>Research reports</h1>
  <p class="muted-text">Every saved run, newest first. Click a date to open that report.
     Change is the score difference from the previous run of the same ticker, only
     between runs of the same model version; a version break marks where it changed.</p>
</header>
{body}
<footer>
  Research output. Not investment advice. No orders are placed.
  <div class="brand-footer">nova-research &middot; Nikolay Gelshtein</div>
</footer>
</div></body></html>"""


def write_index(directory: str | Path = "reports") -> Path:
    path = Path(directory) / INDEX_FILE
    atomic_write_text(path, render_index(directory))
    return path
