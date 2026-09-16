"""Structured logging and run/agent/skill/tool attribution.

Every operation in the system emits an event with the same shape, so the full chain

    research -> source -> synthesis -> verdict -> draft -> citation
             -> verification -> QA -> git -> email

is reconstructable by filtering on ``run_id`` or ``research_id``.

Attribution is carried in ``contextvars`` rather than passed explicitly, so a tool
deep inside a skill still records which agent invoked it without every function
signature growing a parameter.
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import get_config, get_redactor

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
_research_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("research_id", default=None)
_pipeline_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("pipeline_id", default=None)
_agent: contextvars.ContextVar[str | None] = contextvars.ContextVar("agent", default=None)
_skill: contextvars.ContextVar[str | None] = contextvars.ContextVar("skill", default=None)
_state: contextvars.ContextVar[str | None] = contextvars.ContextVar("state", default=None)

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

_CONSOLE_COLOR = {
    "DEBUG": "\033[90m",
    "INFO": "\033[0m",
    "WARN": "\033[33m",
    "ERROR": "\033[31m",
}
_RESET = "\033[0m"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class EventLogger:
    """Writes newline-delimited JSON events plus a readable console stream."""

    def __init__(self) -> None:
        cfg = get_config()
        self.log_dir: Path = cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console: bool = bool(cfg.get("observability.console", True))
        self.jsonl: bool = bool(cfg.get("observability.jsonl", True))
        self.min_level: int = _LEVELS.get(str(cfg.get("observability.log_level", "INFO")).upper(), 20)
        self._redact = get_redactor()
        self._lock = threading.Lock()
        self._sinks: list[Any] = []

    def add_sink(self, fn: Any) -> None:
        """Register an extra consumer (the DB event mirror uses this)."""
        self._sinks.append(fn)

    def _path_for_today(self) -> Path:
        return self.log_dir / f"events-{datetime.now(timezone.utc):%Y%m%d}.jsonl"

    def emit(
        self,
        message: str,
        *,
        level: str = "INFO",
        tool: str | None = None,
        status: str | None = None,
        duration_ms: int | None = None,
        artifact: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": now_iso(),
            "level": level,
            "run_id": _run_id.get(),
            "research_id": _research_id.get(),
            "pipeline_id": _pipeline_id.get(),
            "agent": _agent.get(),
            "skill": _skill.get(),
            "tool": tool,
            "state": _state.get(),
            "status": status,
            "duration_ms": duration_ms,
            "artifact": artifact,
            "message": message,
        }
        if extra:
            event["extra"] = extra
        event = self._redact.scrub(event)

        if _LEVELS.get(level, 20) >= self.min_level:
            if self.jsonl:
                self._write_jsonl(event)
            if self.console:
                self._write_console(event)
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:  # a broken sink must never break the pipeline
                pass
        return event

    def _write_jsonl(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with self._path_for_today().open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _write_console(self, event: dict[str, Any]) -> None:
        parts = [f"{event['ts'][11:23]}"]
        who = event.get("agent") or "system"
        if event.get("skill"):
            who = f"{who}/{event['skill']}"
        if event.get("tool"):
            who = f"{who}:{event['tool']}"
        parts.append(f"[{who}]")
        if event.get("state"):
            parts.append(f"<{event['state']}>")
        parts.append(str(event.get("message", "")))
        if event.get("status"):
            parts.append(f"({event['status']})")
        if event.get("duration_ms") is not None:
            parts.append(f"{event['duration_ms']}ms")
        color = _CONSOLE_COLOR.get(event.get("level", "INFO"), "")
        stream = sys.stderr if event.get("level") in ("WARN", "ERROR") else sys.stdout
        try:
            stream.write(f"{color}{' '.join(parts)}{_RESET}\n")
            stream.flush()
        except Exception:  # pragma: no cover - closed stream during shutdown
            pass

    # convenience -----------------------------------------------------------
    def debug(self, msg: str, **kw: Any) -> dict[str, Any]:
        return self.emit(msg, level="DEBUG", **kw)

    def info(self, msg: str, **kw: Any) -> dict[str, Any]:
        return self.emit(msg, level="INFO", **kw)

    def warn(self, msg: str, **kw: Any) -> dict[str, Any]:
        return self.emit(msg, level="WARN", **kw)

    def error(self, msg: str, **kw: Any) -> dict[str, Any]:
        return self.emit(msg, level="ERROR", **kw)


_logger: EventLogger | None = None


def get_logger() -> EventLogger:
    global _logger
    if _logger is None:
        _logger = EventLogger()
    return _logger


# -- attribution scopes ----------------------------------------------------


@contextmanager
def run_scope(run_id: str | None = None) -> Iterator[str]:
    rid = run_id or new_id("run")
    token = _run_id.set(rid)
    try:
        yield rid
    finally:
        _run_id.reset(token)


@contextmanager
def agent_scope(agent: str) -> Iterator[None]:
    token = _agent.set(agent)
    try:
        yield
    finally:
        _agent.reset(token)


@contextmanager
def skill_scope(skill: str) -> Iterator[None]:
    token = _skill.set(skill)
    try:
        yield
    finally:
        _skill.reset(token)


@contextmanager
def pipeline_scope(pipeline_id: str | None, state: str | None = None) -> Iterator[None]:
    t1 = _pipeline_id.set(pipeline_id)
    t2 = _state.set(state) if state else None
    try:
        yield
    finally:
        _pipeline_id.reset(t1)
        if t2 is not None:
            _state.reset(t2)


@contextmanager
def research_scope(research_id: str | None) -> Iterator[None]:
    token = _research_id.set(research_id)
    try:
        yield
    finally:
        _research_id.reset(token)


@contextmanager
def state_scope(state: str) -> Iterator[None]:
    token = _state.set(state)
    try:
        yield
    finally:
        _state.reset(token)


@contextmanager
def timed(message: str, *, tool: str | None = None, **extra: Any) -> Iterator[dict[str, Any]]:
    """Log start/end of an operation with its duration and outcome."""
    log = get_logger()
    box: dict[str, Any] = {}
    start = time.monotonic()
    log.debug(f"{message} …", tool=tool, status="start", **extra)
    try:
        yield box
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        log.error(
            f"{message} failed: {type(exc).__name__}: {exc}",
            tool=tool, status="error", duration_ms=elapsed, **extra,
        )
        raise
    else:
        elapsed = int((time.monotonic() - start) * 1000)
        log.info(
            message, tool=tool, status="ok", duration_ms=elapsed,
            artifact=box.get("artifact"), **extra,
        )


def current_attribution() -> dict[str, str | None]:
    return {
        "run_id": _run_id.get(),
        "research_id": _research_id.get(),
        "pipeline_id": _pipeline_id.get(),
        "agent": _agent.get(),
        "skill": _skill.get(),
        "state": _state.get(),
    }
