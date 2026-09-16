"""Pipeline state machine.

Transitions are a table, not scattered ``if`` statements. An illegal transition raises
rather than silently corrupting a pipeline, and every legal transition is persisted in
the same transaction as the artifact that justified it — which is what makes a crash
resumable rather than merely survivable.

Backward edges exist (verification sending work back to the writer) but are bounded by
``qa.max_revisions``; exceeding it escalates to NEEDS_HUMAN instead of looping forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import get_config
from ..obs import get_logger, new_id
from .store import Store, get_store, utcnow


class State(str, Enum):
    DISCOVERED = "DISCOVERED"
    COLLECTED = "COLLECTED"
    DEDUPLICATED = "DEDUPLICATED"
    SYNTHESIZED = "SYNTHESIZED"
    VERDICT_PENDING = "VERDICT_PENDING"
    VERDICT_APPROVED = "VERDICT_APPROVED"
    DRAFTED = "DRAFTED"
    TECHNICAL_VERIFY = "TECHNICAL_VERIFY"
    CITATION_AUDIT = "CITATION_AUDIT"
    CITATION_FIND = "CITATION_FIND"
    CITATION_VERIFY = "CITATION_VERIFY"
    EDITORIAL_VERIFY = "EDITORIAL_VERIFY"
    VISUAL = "VISUAL"
    BOOK_QA = "BOOK_QA"
    APPROVED = "APPROVED"
    GIT_PUBLISH = "GIT_PUBLISH"
    EMAIL = "EMAIL"
    COMPLETED = "COMPLETED"

    # off-ramps
    MONITORING = "MONITORING"
    REJECTED = "REJECTED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    FAILED = "FAILED"


TERMINAL = {State.COMPLETED, State.REJECTED, State.FAILED}
PARKED = {State.MONITORING, State.NEEDS_HUMAN}

#: The happy path, in order. Used to drive "advance one step".
FORWARD: list[State] = [
    State.DISCOVERED,
    State.COLLECTED,
    State.DEDUPLICATED,
    State.SYNTHESIZED,
    State.VERDICT_PENDING,
    State.VERDICT_APPROVED,
    State.DRAFTED,
    State.TECHNICAL_VERIFY,
    State.CITATION_AUDIT,
    State.CITATION_FIND,
    State.CITATION_VERIFY,
    State.EDITORIAL_VERIFY,
    State.VISUAL,
    State.BOOK_QA,
    State.APPROVED,
    State.GIT_PUBLISH,
    State.EMAIL,
    State.COMPLETED,
]

_ANYWHERE = {State.FAILED, State.NEEDS_HUMAN}

TRANSITIONS: dict[State, set[State]] = {
    State.DISCOVERED:       {State.COLLECTED, State.REJECTED} | _ANYWHERE,
    State.COLLECTED:        {State.DEDUPLICATED, State.REJECTED} | _ANYWHERE,
    State.DEDUPLICATED:     {State.SYNTHESIZED, State.REJECTED} | _ANYWHERE,
    State.SYNTHESIZED:      {State.VERDICT_PENDING, State.REJECTED} | _ANYWHERE,
    # The gate: a verdict may send work forward, park it, or drop it.
    State.VERDICT_PENDING:  {State.VERDICT_APPROVED, State.MONITORING, State.REJECTED} | _ANYWHERE,
    State.VERDICT_APPROVED: {State.DRAFTED, State.REJECTED} | _ANYWHERE,
    State.DRAFTED:          {State.TECHNICAL_VERIFY} | _ANYWHERE,
    State.TECHNICAL_VERIFY: {State.CITATION_AUDIT, State.DRAFTED} | _ANYWHERE,
    State.CITATION_AUDIT:   {State.CITATION_FIND, State.EDITORIAL_VERIFY} | _ANYWHERE,
    State.CITATION_FIND:    {State.CITATION_VERIFY, State.DRAFTED} | _ANYWHERE,
    State.CITATION_VERIFY:  {State.EDITORIAL_VERIFY, State.CITATION_FIND, State.DRAFTED} | _ANYWHERE,
    State.EDITORIAL_VERIFY: {State.VISUAL, State.DRAFTED} | _ANYWHERE,
    State.VISUAL:           {State.BOOK_QA} | _ANYWHERE,
    State.BOOK_QA:          {State.APPROVED, State.DRAFTED} | _ANYWHERE,
    State.APPROVED:         {State.GIT_PUBLISH} | _ANYWHERE,
    State.GIT_PUBLISH:      {State.EMAIL, State.COMPLETED} | _ANYWHERE,
    State.EMAIL:            {State.COMPLETED} | _ANYWHERE,
    State.COMPLETED:        set(),
    State.MONITORING:       {State.VERDICT_PENDING, State.REJECTED} | _ANYWHERE,
    State.REJECTED:         set(),
    # A human can resume a parked pipeline from where it stopped.
    State.NEEDS_HUMAN:      set(State) - {State.COMPLETED},
    State.FAILED:           {State.DISCOVERED, State.VERDICT_PENDING, State.DRAFTED},
}

#: States that send work backwards, and therefore count against max_revisions.
BACKWARD_EDGES = {
    (State.TECHNICAL_VERIFY, State.DRAFTED),
    (State.CITATION_VERIFY, State.CITATION_FIND),
    (State.CITATION_VERIFY, State.DRAFTED),
    (State.EDITORIAL_VERIFY, State.DRAFTED),
    (State.BOOK_QA, State.DRAFTED),
    (State.CITATION_FIND, State.DRAFTED),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class Pipeline:
    id: str
    cluster_id: str | None
    verdict_id: str | None
    state: State
    previous_state: State | None
    attempts: int
    revisions: int
    last_error: str | None
    data: dict[str, Any]
    run_id: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: Any) -> "Pipeline":
        return cls(
            id=row["id"],
            cluster_id=row["cluster_id"],
            verdict_id=row["verdict_id"],
            state=State(row["state"]),
            previous_state=State(row["previous_state"]) if row["previous_state"] else None,
            attempts=row["attempts"] or 0,
            revisions=row["revisions"] or 0,
            last_error=row["last_error"],
            data=json.loads(row["state_data_json"] or "{}"),
            run_id=row["run_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def is_parked(self) -> bool:
        return self.state in PARKED


class StateMachine:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or get_store()
        self.log = get_logger()
        self.max_revisions = int(get_config().get("qa.max_revisions", 2))

    # -- lifecycle ---------------------------------------------------------
    def create(self, *, cluster_id: str | None = None, run_id: str | None = None,
               state: State = State.SYNTHESIZED, data: dict[str, Any] | None = None) -> Pipeline:
        pid = new_id("pipe")
        now = utcnow()
        self.store.execute(
            "INSERT INTO pipelines (id, cluster_id, state, attempts, revisions, "
            "state_data_json, run_id, created_at, updated_at) VALUES (?,?,?,0,0,?,?,?,?)",
            (pid, cluster_id, state.value, json.dumps(data or {}, default=str), run_id, now, now),
        )
        self.store.execute(
            "INSERT INTO pipeline_transitions (pipeline_id, from_state, to_state, at, note) "
            "VALUES (?,?,?,?,?)",
            (pid, None, state.value, now, "created"),
        )
        self.log.info(f"pipeline created at {state.value}", extra_pipeline=pid)
        return self.get(pid)

    def get(self, pipeline_id: str) -> Pipeline:
        row = self.store.query_one("SELECT * FROM pipelines WHERE id = ?", (pipeline_id,))
        if not row:
            raise KeyError(f"No pipeline {pipeline_id!r}")
        return Pipeline.from_row(row)

    def active(self, limit: int = 100) -> list[Pipeline]:
        """Pipelines that a tick can move — excludes terminal and parked ones."""
        excluded = [s.value for s in TERMINAL | PARKED]
        placeholders = ",".join("?" * len(excluded))
        rows = self.store.query(
            f"SELECT * FROM pipelines WHERE state NOT IN ({placeholders}) "
            "ORDER BY updated_at ASC LIMIT ?",
            (*excluded, limit),
        )
        return [Pipeline.from_row(r) for r in rows]

    def in_state(self, state: State, limit: int = 100) -> list[Pipeline]:
        rows = self.store.query(
            "SELECT * FROM pipelines WHERE state = ? ORDER BY updated_at ASC LIMIT ?",
            (state.value, limit),
        )
        return [Pipeline.from_row(r) for r in rows]

    # -- transitions -------------------------------------------------------
    def can_transition(self, frm: State, to: State) -> bool:
        return to in TRANSITIONS.get(frm, set())

    def transition(
        self,
        pipeline: Pipeline | str,
        to: State,
        *,
        note: str = "",
        data: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Pipeline:
        """Move a pipeline, persisting state and artifact in one transaction."""
        pipe = self.get(pipeline) if isinstance(pipeline, str) else pipeline

        if not self.can_transition(pipe.state, to):
            raise IllegalTransition(
                f"{pipe.id}: {pipe.state.value} -> {to.value} is not a legal transition"
            )

        revisions = pipe.revisions
        if (pipe.state, to) in BACKWARD_EDGES:
            revisions += 1
            if revisions > self.max_revisions:
                self.log.warn(
                    f"{pipe.id}: revision limit ({self.max_revisions}) exceeded at "
                    f"{pipe.state.value}; escalating to human review"
                )
                to = State.NEEDS_HUMAN
                note = (note + " | " if note else "") + (
                    f"exceeded max_revisions={self.max_revisions}"
                )

        merged = dict(pipe.data)
        if data:
            merged.update(data)
        now = utcnow()

        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE pipelines SET state=?, previous_state=?, revisions=?, "
                "state_data_json=?, updated_at=?, last_error=? WHERE id=?",
                (to.value, pipe.state.value, revisions,
                 json.dumps(merged, default=str), now,
                 None if to != State.FAILED else pipe.last_error, pipe.id),
            )
            conn.execute(
                "INSERT INTO pipeline_transitions "
                "(pipeline_id, from_state, to_state, at, note, artifact_id) VALUES (?,?,?,?,?,?)",
                (pipe.id, pipe.state.value, to.value, now, note or None, artifact_id),
            )

        self.log.info(
            f"{pipe.state.value} -> {to.value}" + (f" ({note})" if note else ""),
            status="transition", artifact=artifact_id,
        )
        return self.get(pipe.id)

    def advance(self, pipeline: Pipeline, **kw: Any) -> Pipeline:
        """Move one step along the happy path."""
        try:
            idx = FORWARD.index(pipeline.state)
        except ValueError:
            raise IllegalTransition(f"{pipeline.state.value} is not on the forward path")
        if idx + 1 >= len(FORWARD):
            raise IllegalTransition(f"{pipeline.state.value} is the final forward state")
        return self.transition(pipeline, FORWARD[idx + 1], **kw)

    def fail(self, pipeline: Pipeline | str, error: str, *, note: str = "") -> Pipeline:
        pipe = self.get(pipeline) if isinstance(pipeline, str) else pipeline
        self.store.execute(
            "UPDATE pipelines SET last_error=?, attempts=attempts+1 WHERE id=?",
            (error[:2000], pipe.id),
        )
        pipe = self.get(pipe.id)
        return self.transition(pipe, State.FAILED, note=note or error[:200])

    def park_for_human(self, pipeline: Pipeline | str, reason: str) -> Pipeline:
        pipe = self.get(pipeline) if isinstance(pipeline, str) else pipeline
        return self.transition(pipe, State.NEEDS_HUMAN, note=reason)

    def record_attempt(self, pipeline: Pipeline) -> None:
        self.store.execute(
            "UPDATE pipelines SET attempts=attempts+1, updated_at=? WHERE id=?",
            (utcnow(), pipeline.id),
        )

    def history(self, pipeline_id: str) -> list[Any]:
        return self.store.query(
            "SELECT * FROM pipeline_transitions WHERE pipeline_id=? ORDER BY id",
            (pipeline_id,),
        )

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, int]:
        rows = self.store.query(
            "SELECT state, COUNT(*) AS n FROM pipelines GROUP BY state ORDER BY n DESC"
        )
        return {r["state"]: r["n"] for r in rows}
