"""Shared pytest configuration.

Redirects logging and state away from the live directories. Without this, running the
suite writes test pipelines into `logs/events-*.jsonl` and an operator reading the
unattended log sees transitions that never happened in production.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="livingbook_tests_"))


def pytest_configure(config):  # noqa: ARG001
    """Point observability at a throwaway directory before anything imports it."""
    from livingbook import config as lb_config

    cfg = lb_config.get_config()
    cfg.data.setdefault("observability", {})
    cfg.data["observability"]["log_dir"] = str(_TMP / "logs")
    cfg.data["observability"]["console"] = False

    # get_logger() caches an EventLogger bound to the log dir, so it must be reset
    # after the override rather than before.
    from livingbook import obs

    obs._logger = None
