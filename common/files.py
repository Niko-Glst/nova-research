"""Write files so an interruption never leaves half a file behind.

Caches are trusted by age: a price CSV cut short by a crash, a full disk or the
PC going to sleep would otherwise be read back as fresh history with its recent
rows missing. Writing to a temporary file in the same folder and swapping it in
with os.replace means a reader sees either the old file or the new one, whole.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: str | Path, write: Callable[[Path], object]) -> Path:
    """Call `write(temporary_path)`, then move the result over `path` in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # the write or the swap failed
            temporary.unlink()
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write(path, lambda p: p.write_text(text, encoding=encoding))
