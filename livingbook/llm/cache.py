"""Content-addressed response cache.

The obvious benefit is quota, but the load-bearing one is resumability: when a
pipeline crashes at, say, CITATION_VERIFY and restarts, every completion the earlier
states already produced replays from cache instead of being re-billed and re-sampled.
That is what makes "resume, don't restart" affordable.

Keyed on sha256(model, payload) where payload includes the schema and generation
config, so a temperature or schema change is a different entry rather than a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    hash         TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    kind         TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_created ON llm_cache(created_at);
"""


def cache_key(model: str, kind: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"model": model, "kind": kind, "payload": payload},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path, *, ttl_days: int = 30, enabled: bool = True) -> None:
        self.path = path
        self.ttl_seconds = ttl_days * 86400
        self.enabled = enabled
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connect()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT response_json, created_at FROM llm_cache WHERE hash = ?", (key,)
            ).fetchone()
        if not row:
            return None
        response_json, created_at = row
        if self.ttl_seconds and time.time() - created_at > self.ttl_seconds:
            self.delete(key)
            return None
        try:
            return json.loads(response_json)
        except Exception:
            return None

    def put(self, key: str, model: str, kind: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            blob = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (hash, model, kind, response_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, model, kind, blob, time.time()),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM llm_cache WHERE hash = ?", (key,))
            conn.commit()

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "entries": 0}
        with self._lock:
            conn = self._connect()
            n = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            by_model = conn.execute(
                "SELECT model, COUNT(*) FROM llm_cache GROUP BY model ORDER BY 2 DESC"
            ).fetchall()
        return {"enabled": True, "entries": n, "by_model": dict(by_model)}

    def purge_expired(self) -> int:
        if not self.enabled or not self.ttl_seconds:
            return 0
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            conn = self._connect()
            cur = conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
