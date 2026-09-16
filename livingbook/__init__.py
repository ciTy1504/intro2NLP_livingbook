"""Living Book — an autonomous research-to-publication pipeline for an NLP textbook.

Importing the package forces UTF-8 on stdio. The manuscript is Vietnamese, so on a
Windows console (cp1252 by default) every log line containing a chapter title would
otherwise raise UnicodeEncodeError and take the process down with it.
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"


def _force_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached stream
                pass


_force_utf8_stdio()
