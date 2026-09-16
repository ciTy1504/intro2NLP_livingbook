"""API key pool with rotation, cooldown, eviction and usage tracking.

Adapted from ``cti-realm-sft-final/scripts/common/gemini_pool.py`` in the LLM project.
Two behaviours are carried over deliberately, with the original reasoning intact:

1. **Random selection, not round-robin.** The pool is shared across independent
   processes. A round-robin cursor starts at 0 in every process, so concurrent
   processes reach for key[0], then key[1], in lockstep — guaranteed collisions
   instead of the load-spreading round-robin was meant to provide. Random selection
   decorrelates them with zero cross-process coordination.

2. **Cooldown rather than eviction on 429/403.** A rate-limited key recovers; removing
   it would permanently shrink a pool that is already the system's scarcest resource.

Added here on top of the original: permanent eviction of structurally invalid keys,
per-key usage/token accounting persisted across restarts, and a status view for the
CLI. Keys are identified in all output by a truncated SHA-256 — the plaintext key
never reaches a log line, an artifact or the database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..obs import get_logger


def key_fingerprint(key: str) -> str:
    """Stable, non-reversible identifier for a key, safe to log and persist."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


@dataclass
class KeyStats:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None
    #: Consecutive hard failures with no intervening success. A key is only retired
    #: once this crosses a threshold, so a transient 403 cannot kill a working key.
    strikes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "last_used": self.last_used,
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "strikes": self.strikes,
        }


class NoKeysAvailable(RuntimeError):
    """Every key is disabled (as opposed to merely cooling down)."""


class KeyPool:
    """Async-safe pool of interchangeable API credentials."""

    def __init__(
        self,
        keys: Iterable[str],
        *,
        cooldown_seconds: float = 90.0,
        stats_path: Path | None = None,
        max_strikes: int = 5,
    ) -> None:
        self.keys: list[str] = list(keys)
        if not self.keys:
            raise ValueError("KeyPool needs at least one key")
        self.cooldown_seconds = cooldown_seconds
        self.stats_path = stats_path
        self.max_strikes = max_strikes
        self._stats: dict[str, KeyStats] = {key_fingerprint(k): KeyStats() for k in self.keys}
        self._by_fp: dict[str, str] = {key_fingerprint(k): k for k in self.keys}
        self._lock = asyncio.Lock()
        self._log = get_logger()
        self._load_stats()

    # -- persistence -------------------------------------------------------
    def _load_stats(self) -> None:
        if not self.stats_path or not self.stats_path.exists():
            return
        try:
            saved = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for fp, blob in saved.items():
            if fp not in self._stats:
                continue  # a key that has since left the pool
            st = self._stats[fp]
            st.requests = int(blob.get("requests", 0))
            st.successes = int(blob.get("successes", 0))
            st.failures = int(blob.get("failures", 0))
            st.tokens_in = int(blob.get("tokens_in", 0))
            st.tokens_out = int(blob.get("tokens_out", 0))
            st.last_used = float(blob.get("last_used", 0.0))
            # Disabled state survives a restart; cooldowns do not (they are short
            # and wall-clock based, so re-testing after a restart is correct).
            st.disabled = bool(blob.get("disabled", False))
            st.disable_reason = blob.get("disable_reason")

    def save_stats(self) -> None:
        if not self.stats_path:
            return
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {fp: st.as_dict() for fp, st in self._stats.items()}
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.stats_path)
        except Exception as exc:  # never let bookkeeping break a request
            self._log.debug(f"keypool stats not saved: {exc}")

    # -- selection ---------------------------------------------------------
    async def acquire(self, *, exclude: set[str] | None = None, max_wait: float = 120.0) -> str:
        """Return an available key, waiting for the soonest cooldown if all are busy.

        ``exclude`` holds fingerprints already tried for the current request, so a
        retry loop never picks the same key twice in a row.
        """
        exclude = exclude or set()
        deadline = time.monotonic() + max_wait

        while True:
            async with self._lock:
                now = time.monotonic()
                live = [
                    fp for fp, st in self._stats.items()
                    if not st.disabled and fp not in exclude
                ]
                if not live:
                    if all(st.disabled for st in self._stats.values()):
                        raise NoKeysAvailable(
                            f"all {len(self.keys)} keys are disabled; "
                            "check secrets/gemini_keypool.txt"
                        )
                    # Everything live is excluded for this request: allow reuse.
                    live = [fp for fp, st in self._stats.items() if not st.disabled]
                    exclude = set()

                ready = [fp for fp in live if self._stats[fp].cooldown_until <= now]
                if ready:
                    fp = random.choice(ready)
                    st = self._stats[fp]
                    st.requests += 1
                    st.last_used = time.time()
                    return self._by_fp[fp]

                soonest = min(self._stats[fp].cooldown_until for fp in live)
                wait = max(0.25, soonest - now)

            if time.monotonic() + wait > deadline:
                wait = max(0.25, deadline - time.monotonic())
                if wait <= 0.25:
                    raise NoKeysAvailable(
                        f"every key cooling down; waited {max_wait:.0f}s"
                    )
            self._log.debug(f"keypool: all keys cooling, waiting {wait:.1f}s")
            await asyncio.sleep(min(wait, 10.0))

    # -- feedback ----------------------------------------------------------
    async def report_success(self, key: str, *, tokens_in: int = 0, tokens_out: int = 0) -> None:
        fp = key_fingerprint(key)
        async with self._lock:
            st = self._stats.get(fp)
            if st:
                st.successes += 1
                st.tokens_in += tokens_in
                st.tokens_out += tokens_out
                st.strikes = 0  # a success clears the strike record

    async def report_cooldown(self, key: str, *, seconds: float | None = None, reason: str = "") -> None:
        """Take a key out of rotation temporarily (429 / quota)."""
        fp = key_fingerprint(key)
        async with self._lock:
            st = self._stats.get(fp)
            if st:
                st.failures += 1
                st.cooldown_until = time.monotonic() + (seconds or self.cooldown_seconds)
        self._log.debug(f"keypool: key {fp} cooling down ({reason or 'rate limited'})")

    async def report_invalid(self, key: str, *, reason: str = "invalid",
                             immediate: bool = False) -> None:
        """Record a hard credential failure.

        ``immediate`` is reserved for structurally dead credentials (a 400
        API_KEY_INVALID). Everything else accumulates strikes and only retires the key
        after ``max_strikes`` consecutive failures with no success in between —
        retiring on a single response would shrink the pool over transient errors,
        which is exactly what happened before this guard existed.
        """
        fp = key_fingerprint(key)
        async with self._lock:
            st = self._stats.get(fp)
            if not st or st.disabled:
                return
            st.failures += 1
            st.strikes += 1
            if immediate or st.strikes >= self.max_strikes:
                st.disabled = True
                st.disable_reason = f"{reason} (after {st.strikes} strikes)"
                remaining = sum(1 for s in self._stats.values() if not s.disabled)
                self._log.warn(
                    f"keypool: key {fp} disabled ({st.disable_reason}); "
                    f"{remaining} keys remain"
                )
                should_save = True
            else:
                st.cooldown_until = time.monotonic() + self.cooldown_seconds
                should_save = False
        if should_save:
            self.save_stats()

    async def revive_disabled(self) -> int:
        """Clear every disable, so a pool damaged by a transient outage recovers.

        Exposed on the CLI (`llm revive`) because a provider-side incident can retire
        keys that are in fact fine, and there is otherwise no way back.
        """
        revived = 0
        async with self._lock:
            for st in self._stats.values():
                if st.disabled:
                    st.disabled = False
                    st.disable_reason = None
                    st.strikes = 0
                    st.cooldown_until = 0.0
                    revived += 1
        self.save_stats()
        if revived:
            self._log.info(f"keypool: revived {revived} disabled keys")
        return revived

    async def report_failure(self, key: str) -> None:
        fp = key_fingerprint(key)
        async with self._lock:
            st = self._stats.get(fp)
            if st:
                st.failures += 1

    # -- introspection -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        available = sum(
            1 for st in self._stats.values()
            if not st.disabled and st.cooldown_until <= now
        )
        cooling = sum(
            1 for st in self._stats.values()
            if not st.disabled and st.cooldown_until > now
        )
        disabled = sum(1 for st in self._stats.values() if st.disabled)
        return {
            "total": len(self.keys),
            "available": available,
            "cooling_down": cooling,
            "disabled": disabled,
            "requests": sum(st.requests for st in self._stats.values()),
            "successes": sum(st.successes for st in self._stats.values()),
            "failures": sum(st.failures for st in self._stats.values()),
            "tokens_in": sum(st.tokens_in for st in self._stats.values()),
            "tokens_out": sum(st.tokens_out for st in self._stats.values()),
        }

    def per_key_status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        rows = []
        for fp, st in self._stats.items():
            rows.append({
                "key": fp,
                "requests": st.requests,
                "successes": st.successes,
                "failures": st.failures,
                "tokens": st.tokens_in + st.tokens_out,
                "state": (
                    "disabled" if st.disabled
                    else "cooling" if st.cooldown_until > now
                    else "ready"
                ),
                "reason": st.disable_reason,
            })
        return sorted(rows, key=lambda r: (-r["requests"], r["key"]))
