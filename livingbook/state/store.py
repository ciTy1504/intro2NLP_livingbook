"""SQLite-backed persistent store.

One connection per process, WAL, with a ``transaction()`` context manager. Everything
that must be atomic — notably "advance the pipeline state *and* record the artifact
that justified it" — goes through that single transaction, which is what makes
resume-after-crash correct rather than merely likely.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..config import get_config
from ..obs import new_id

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class Store:
    """Thin, explicit data-access layer. No ORM, no magic."""

    def __init__(self, path: Path | None = None) -> None:
        cfg = get_config()
        self.path = Path(path) if path else cfg.state_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._init_schema()

    # -- connection --------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                pass  # already inside a transaction
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- primitives --------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._write_lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row else None

    # -- runs --------------------------------------------------------------
    def start_run(self, run_id: str, trigger: str = "manual", config_hash: str = "") -> str:
        self.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, status, trigger, config_hash) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, utcnow(), trigger, config_hash),
        )
        return run_id

    def end_run(self, run_id: str, status: str = "completed", summary: dict[str, Any] | None = None) -> None:
        self.execute(
            "UPDATE runs SET ended_at = ?, status = ?, summary_json = ? WHERE run_id = ?",
            (utcnow(), status, json.dumps(summary or {}, default=str), run_id),
        )

    # -- events ------------------------------------------------------------
    def record_event(self, event: dict[str, Any]) -> None:
        """Mirror a structured log event into the DB. Registered as a logger sink."""
        try:
            with self._write_lock:
                self.conn.execute(
                    "INSERT INTO events (ts, level, run_id, research_id, pipeline_id, agent, "
                    "skill, tool, state, status, duration_ms, artifact, message, extra_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event.get("ts"), event.get("level"), event.get("run_id"),
                        event.get("research_id"), event.get("pipeline_id"),
                        event.get("agent"), event.get("skill"), event.get("tool"),
                        event.get("state"), event.get("status"), event.get("duration_ms"),
                        event.get("artifact"), event.get("message"),
                        json.dumps(event.get("extra"), default=str) if event.get("extra") else None,
                    ),
                )
                self.conn.commit()
        except Exception:
            # Logging must never be able to break the thing it is observing.
            pass

    def trace(self, *, run_id: str | None = None, research_id: str | None = None,
              pipeline_id: str | None = None, limit: int = 500) -> list[sqlite3.Row]:
        where, params = [], []
        if run_id:
            where.append("run_id = ?"); params.append(run_id)
        if research_id:
            where.append("research_id = ?"); params.append(research_id)
        if pipeline_id:
            where.append("pipeline_id = ?"); params.append(pipeline_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        return self.query(f"SELECT * FROM events {clause} ORDER BY id ASC LIMIT ?", params)

    # -- embeddings --------------------------------------------------------
    def put_embedding(self, owner_type: str, owner_id: str, model: str, vector: Sequence[float]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO embeddings (owner_type, owner_id, model, dim, vector, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (owner_type, owner_id, model, len(vector), pack_vector(vector), utcnow()),
        )

    def put_embeddings(self, rows: Iterable[tuple[str, str, str, Sequence[float]]]) -> None:
        now = utcnow()
        self.executemany(
            "INSERT OR REPLACE INTO embeddings (owner_type, owner_id, model, dim, vector, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(ot, oid, m, len(v), pack_vector(v), now) for ot, oid, m, v in rows],
        )

    def get_embedding(self, owner_type: str, owner_id: str) -> list[float] | None:
        row = self.query_one(
            "SELECT vector FROM embeddings WHERE owner_type = ? AND owner_id = ? LIMIT 1",
            (owner_type, owner_id),
        )
        return unpack_vector(row["vector"]) if row else None

    def all_embeddings(self, owner_type: str) -> list[tuple[str, list[float]]]:
        rows = self.query(
            "SELECT owner_id, vector FROM embeddings WHERE owner_type = ?", (owner_type,)
        )
        return [(r["owner_id"], unpack_vector(r["vector"])) for r in rows]

    def has_embedding(self, owner_type: str, owner_id: str) -> bool:
        return self.scalar(
            "SELECT 1 FROM embeddings WHERE owner_type = ? AND owner_id = ? LIMIT 1",
            (owner_type, owner_id),
        ) is not None

    # -- graph -------------------------------------------------------------
    def add_edge(self, src_type: str, src_id: str, rel: str, dst_type: str, dst_id: str,
                 *, weight: float = 1.0, provenance: dict[str, Any] | None = None) -> None:
        self.execute(
            "INSERT OR REPLACE INTO graph_edges "
            "(src_type, src_id, rel, dst_type, dst_id, weight, provenance_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (src_type, src_id, rel, dst_type, dst_id, weight,
             json.dumps(provenance, default=str) if provenance else None, utcnow()),
        )

    def add_edges(self, edges: Iterable[tuple]) -> None:
        now = utcnow()
        self.executemany(
            "INSERT OR REPLACE INTO graph_edges "
            "(src_type, src_id, rel, dst_type, dst_id, weight, provenance_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(e[0], e[1], e[2], e[3], e[4],
              e[5] if len(e) > 5 else 1.0,
              json.dumps(e[6], default=str) if len(e) > 6 and e[6] else None,
              now) for e in edges],
        )

    def out_edges(self, src_type: str, src_id: str, rel: str | None = None) -> list[sqlite3.Row]:
        if rel:
            return self.query(
                "SELECT * FROM graph_edges WHERE src_type=? AND src_id=? AND rel=?",
                (src_type, src_id, rel))
        return self.query(
            "SELECT * FROM graph_edges WHERE src_type=? AND src_id=?", (src_type, src_id))

    def in_edges(self, dst_type: str, dst_id: str, rel: str | None = None) -> list[sqlite3.Row]:
        if rel:
            return self.query(
                "SELECT * FROM graph_edges WHERE dst_type=? AND dst_id=? AND rel=?",
                (dst_type, dst_id, rel))
        return self.query(
            "SELECT * FROM graph_edges WHERE dst_type=? AND dst_id=?", (dst_type, dst_id))

    # -- artifacts ---------------------------------------------------------
    def add_artifact(self, *, kind: str, pipeline_id: str | None = None, path: str | None = None,
                     sha256: str | None = None, parent_artifact_id: str | None = None,
                     meta: dict[str, Any] | None = None, created_by: str | None = None) -> str:
        aid = new_id("art")
        self.execute(
            "INSERT INTO artifacts (id, pipeline_id, kind, path, sha256, parent_artifact_id, "
            "meta_json, created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, pipeline_id, kind, path, sha256, parent_artifact_id,
             json.dumps(meta or {}, default=str), utcnow(), created_by),
        )
        return aid

    def artifact_chain(self, artifact_id: str) -> list[sqlite3.Row]:
        """Walk parent links back to the root — the provenance chain for one change."""
        chain: list[sqlite3.Row] = []
        current: str | None = artifact_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            row = self.query_one("SELECT * FROM artifacts WHERE id = ?", (current,))
            if not row:
                break
            chain.append(row)
            current = row["parent_artifact_id"]
        return chain

    def artifacts_for(self, pipeline_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind:
            return self.query(
                "SELECT * FROM artifacts WHERE pipeline_id=? AND kind=? ORDER BY created_at",
                (pipeline_id, kind))
        return self.query(
            "SELECT * FROM artifacts WHERE pipeline_id=? ORDER BY created_at", (pipeline_id,))

    # -- stats -------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        tables = [
            "runs", "events", "research_items", "evidence", "clusters", "book_nodes",
            "node_summaries", "concepts", "claims", "bib_entries", "cite_edges",
            "figures", "graph_edges", "verdicts", "pipelines", "artifacts",
            "versions", "emails_sent", "embeddings",
        ]
        out: dict[str, int] = {}
        for t in tables:
            try:
                out[t] = int(self.scalar(f"SELECT COUNT(*) FROM {t}") or 0)
            except sqlite3.Error:
                out[t] = -1
        return out


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
        # Mirror structured log events into the DB so a trace can be queried by SQL.
        from ..obs import get_logger
        get_logger().add_sink(_store.record_event)
    return _store


def reset_store() -> None:
    global _store
    if _store is not None:
        _store.close()
    _store = None
