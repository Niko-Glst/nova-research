"""Write files so an interruption never leaves half a file behind.

Caches are trusted by age: a price CSV cut short by a crash, a full disk or the
PC going to sleep would otherwise be read back as fresh history with its recent
rows missing. Writing to a temporary file in the same folder and swapping it in
with os.replace means a reader sees either the old file or the new one, whole.

Windows refuses a swap or an append while another process holds the file for a
moment: OneDrive syncing it (this project lives in OneDrive), an antivirus scan,
the search indexer. That shows up as PermissionError ("Toegang geweigerd") and
used to cost a run its snapshot. Such writes are retried with a growing wait.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

LOCK_ATTEMPTS = 6
LOCK_FIRST_WAIT_SECONDS = 0.25  # doubling: about 8 seconds before giving up


def retry_on_lock(action: Callable[[], T], attempts: int = LOCK_ATTEMPTS,
                  first_wait: float = LOCK_FIRST_WAIT_SECONDS,
                  sleep: Callable[[float], object] = time.sleep) -> T:
    """Run `action`, retrying while Windows reports the file as in use."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            sleep(first_wait * 2 ** attempt)
    raise AssertionError("unreachable")


def atomic_write(path: str | Path, write: Callable[[Path], object]) -> Path:
    """Call `write(temporary_path)`, then move the result over `path` in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        write(temporary)
        retry_on_lock(lambda: os.replace(temporary, path))
    finally:
        if temporary.exists():  # the write or the swap failed
            temporary.unlink()
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write(path, lambda p: p.write_text(text, encoding=encoding))
