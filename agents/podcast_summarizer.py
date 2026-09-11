"""Audio -> transcript -> summary.

Takes an investing-related podcast/audio file, transcribes it, and produces a
condensed summary of the investment-relevant claims made in it. No live trading
implications here — output is research input only.

TODO:
- Choose/implement a transcription backend (e.g. local whisper model or an API,
  credentials for which must come from settings.py / .env, never hardcoded).
- Implement summarization, likely reusing whatever LLM client the rest of the
  pipeline settles on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PodcastSummary:
    source_path: Path
    transcript: str
    summary: str
    key_claims: list[str]


def transcribe(audio_path: Path) -> str:
    """Transcribe an audio file to text.

    TODO: implement.
    """
    raise NotImplementedError


def summarize(transcript: str) -> PodcastSummary:
    """Summarize a transcript into key investment-relevant claims.

    TODO: implement.
    """
    raise NotImplementedError
