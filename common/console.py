"""Console output that survives text the console encoding cannot show.

On Windows, output redirected to a file or pipe is written in the ANSI code page
(cp1252), which has no emoji. One emoji in a scraped Reddit title or headline
used to abort `python research.py AAPL > report.txt` with UnicodeEncodeError
after the whole analysis had run. Such characters now print as "?" instead.
"""

from __future__ import annotations

import sys


def safe_console(*streams) -> None:
    """Replace unencodable characters on stdout/stderr (or the given streams)."""
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # a closed or non-reconfigurable stream
            continue
