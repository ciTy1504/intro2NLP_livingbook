"""Persistent scheduler.

Not ``sleep()`` in a loop. Due times live in the database, so a restart does not reset
every schedule, a window missed while the machine was off fires on next start, and two
processes cannot run the same job at once. For a system meant to run unattended for
weeks, those three properties are the difference between a scheduler and a timer.
"""

from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from ..config import get_config
from ..obs import get_logger
from ..state.store import Store, get_store

JobFn = Callable[[], Awaitable[Any]]

#: A job whose lock is older than this is assumed to have died with its process.
STALE_LOCK_SECONDS = 3 * 3600


class Scheduler:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or get_store()
        self.cfg = get_config()
        self.log = get_logger()
        self.jobs: dict[str, JobFn] = {}
        self.intervals: dict[str, int] = {}
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        #: Jobs running right now in this process, so a tick never starts one twice.
        self._running: dict[str, asyncio.Task] = {}

    # -- registration ------------------------------------------------------
    def register(self, name: str, fn: JobFn, *, interval_seconds: int) -> None:
        self.jobs[name] = fn
        self.intervals[name] = interval_seconds
        row = self.store.query_one("SELECT name FROM scheduled_jobs WHERE name = ?", (name,))
        if row:
            self.store.execute(
                "UPDATE scheduled_jobs SET interval_seconds = ? WHERE name = ?",
                (interval_seconds, name))
        else:
            # A brand-new job is due immediately: on first start everything should run
            # once rather than waiting out a full interval.
            self.store.execute(
                "INSERT INTO scheduled_jobs (name, interval_seconds, next_run_at) "
                "VALUES (?,?,?)", (name, interval_seconds, _iso(datetime.now(timezone.utc))))

    def register_from_config(self, handlers: dict[str, JobFn]) -> None:
        jobs = self.cfg.get("scheduler.jobs", {}) or {}
        for name, spec in jobs.items():
            if name not in handlers:
                self.log.warn(f"scheduler: no handler for configured job {name!r}")
                continue
            self.register(name, handlers[name], interval_seconds=_seconds(spec))

    # -- execution ---------------------------------------------------------
    def due(self) -> list[str]:
        now = _iso(datetime.now(timezone.utc))
        rows = self.store.query(
            "SELECT name, running, lock_at FROM scheduled_jobs "
            "WHERE next_run_at IS NULL OR next_run_at <= ? ORDER BY next_run_at", (now,))
        out = []
        for r in rows:
            if r["name"] not in self.jobs:
                continue
            if r["running"] and not _lock_is_stale(r["lock_at"]):
                continue
            out.append(r["name"])
        return out

    def _acquire(self, name: str) -> bool:
        """Claim a job. Returns False if another process already holds it."""
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT running, lock_at FROM scheduled_jobs WHERE name = ?",
                (name,)).fetchone()
            if row and row["running"] and not _lock_is_stale(row["lock_at"]):
                return False
            conn.execute(
                "UPDATE scheduled_jobs SET running = 1, lock_owner = ?, lock_at = ? "
                "WHERE name = ?",
                (self.owner, _iso(datetime.now(timezone.utc)), name))
        return True

    def _release(self, name: str, *, status: str, error: str = "") -> None:
        now = datetime.now(timezone.utc)
        next_run = now + timedelta(seconds=self.intervals.get(name, 3600))
        self.store.execute(
            "UPDATE scheduled_jobs SET running = 0, lock_owner = NULL, lock_at = NULL, "
            "last_run_at = ?, next_run_at = ?, last_status = ?, last_error = ?, "
            "runs = runs + 1 WHERE name = ?",
            (_iso(now), _iso(next_run), status, error[:1000] or None, name))

    async def run_job(self, name: str) -> Any:
        if name not in self.jobs:
            raise KeyError(f"no such job: {name}")
        if not self._acquire(name):
            self.log.debug(f"scheduler: {name} is already running elsewhere")
            return None

        self.log.info(f"scheduler: running {name}")
        try:
            result = await self.jobs[name]()
        except Exception as exc:
            # One failing job must never stop the scheduler; it reschedules and the
            # next window tries again.
            self.log.error(f"scheduler: {name} failed: {type(exc).__name__}: {exc}")
            self._release(name, status="failed", error=f"{type(exc).__name__}: {exc}")
            return None
        self._release(name, status="ok")
        return result

    async def tick(self) -> list[str]:
        """Start everything currently due, concurrently. Returns the job names started.

        Concurrently, not in sequence. A discovery pass can run for hours — it fans out
        to eight source agents, downloads PDFs and makes hundreds of model calls — and
        running jobs one after another meant it starved everything behind it. Measured:
        `pipeline_tick`, scheduled every 15 minutes, had not run in 3.8 hours because
        `discovery` was still going.

        Each job already holds its own lock, so a long job simply stays out of its own
        way on the next tick while the short ones keep their cadence.
        """
        due = self.due()
        if not due:
            return []

        started: list[str] = []
        for name in due:
            if name in self._running:
                continue
            task = asyncio.create_task(self._run_tracked(name))
            self._running[name] = task
            started.append(name)
        return started

    async def _run_tracked(self, name: str) -> None:
        try:
            await self.run_job(name)
        finally:
            self._running.pop(name, None)

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait for in-flight jobs, for a clean shutdown."""
        tasks = list(self._running.values())
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()

    async def run_forever(self, *, poll_seconds: int = 30) -> None:
        self.log.info(
            f"scheduler started ({len(self.jobs)} jobs, polling every {poll_seconds}s)")
        self._log_schedule()
        while True:
            try:
                started = await self.tick()
                if started:
                    self.log.debug(f"scheduler started: {', '.join(started)}")
            except asyncio.CancelledError:
                self.log.info("scheduler stopping")
                await self.drain()
                raise
            except Exception as exc:
                self.log.error(f"scheduler tick failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(poll_seconds)

    # -- reporting ---------------------------------------------------------
    def status(self) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM scheduled_jobs ORDER BY next_run_at")
        return [dict(r) for r in rows]

    def _log_schedule(self) -> None:
        for job in self.status():
            if job["name"] in self.jobs:
                self.log.info(
                    f"  {job['name']:16s} every {_human(job['interval_seconds']):>8s}  "
                    f"next {job['next_run_at'] or 'now'}")

    def reset(self, name: str | None = None) -> None:
        """Clear locks and make jobs due now — for recovering from a hard kill."""
        now = _iso(datetime.now(timezone.utc))
        if name:
            self.store.execute(
                "UPDATE scheduled_jobs SET running = 0, lock_owner = NULL, "
                "lock_at = NULL, next_run_at = ? WHERE name = ?", (now, name))
        else:
            self.store.execute(
                "UPDATE scheduled_jobs SET running = 0, lock_owner = NULL, "
                "lock_at = NULL, next_run_at = ?", (now,))


def _seconds(spec: dict[str, Any]) -> int:
    if not isinstance(spec, dict):
        return 3600
    if "every_seconds" in spec:
        return int(spec["every_seconds"])
    if "every_minutes" in spec:
        return int(spec["every_minutes"]) * 60
    if "every_hours" in spec:
        return int(spec["every_hours"]) * 3600
    if "every_days" in spec:
        return int(spec["every_days"]) * 86400
    return 3600


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _lock_is_stale(lock_at: str | None) -> bool:
    if not lock_at:
        return True
    try:
        held = datetime.fromisoformat(lock_at)
    except ValueError:
        return True
    if held.tzinfo is None:
        held = held.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - held).total_seconds() > STALE_LOCK_SECONDS


def _human(seconds: int) -> str:
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"
