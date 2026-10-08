"""Tests for the report archive and its index page."""

from __future__ import annotations

from datetime import datetime

from agents import report_archive
from agents.research_report import ResearchReport


def _report(symbol: str, score: float, verdict: str = "positive") -> ResearchReport:
    return ResearchReport(symbol=symbol, verdict=verdict, score=score, confidence=0.75)


class TestSaveReport:
    def test_writes_latest_archive_and_index(self, tmp_path):
        latest = report_archive.save_report(
            _report("TEST", 0.4), tmp_path, now=datetime(2026, 10, 8, 12, 45)
        )
        assert latest == tmp_path / "TEST.html"
        assert (tmp_path / "archive" / "TEST_2026-10-08_1245.html").exists()
        assert (tmp_path / "index.html").exists()

    def test_reruns_keep_earlier_copies(self, tmp_path):
        report_archive.save_report(_report("TEST", 0.1), tmp_path, now=datetime(2026, 10, 1, 9, 0))
        report_archive.save_report(_report("TEST", 0.4), tmp_path, now=datetime(2026, 10, 8, 9, 0))

        runs = report_archive.list_runs(tmp_path)["TEST"]
        assert [r.generated.day for r in runs] == [8, 1]  # newest first
        assert [r.score for r in runs] == [0.4, 0.1]
        assert runs[0].verdict == "POSITIVE"
        assert runs[0].confidence == 75


class TestIndex:
    def test_links_every_run_and_shows_score_change(self, tmp_path):
        report_archive.save_report(_report("TEST", 0.1), tmp_path, now=datetime(2026, 10, 1, 9, 0))
        report_archive.save_report(_report("TEST", 0.4), tmp_path, now=datetime(2026, 10, 8, 9, 0))
        report_archive.save_report(_report("SU.PA", -0.2, "negative"), tmp_path,
                                   now=datetime(2026, 10, 8, 10, 0))

        page = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'href="archive/TEST_2026-10-01_0900.html"' in page
        assert 'href="archive/TEST_2026-10-08_0900.html"' in page
        assert 'href="archive/SU.PA_2026-10-08_1000.html"' in page
        assert "+0.30" in page  # 0.4 vs the previous 0.1
        assert "2 runs" in page and "1 run<" in page

    def test_unrelated_files_are_ignored(self, tmp_path):
        (tmp_path / "archive").mkdir()
        (tmp_path / "archive" / "notes.html").write_text("x", encoding="utf-8")
        assert report_archive.list_runs(tmp_path) == {}

    def test_empty_archive_renders(self, tmp_path):
        assert "No reports yet" in report_archive.render_index(tmp_path)
