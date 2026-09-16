"""Artifact store and provenance chain.

Every step of the pipeline writes a structured artifact to disk and a row linking it to
its parent. That linkage is what makes the requirement

    book change -> verdict -> research cluster -> evidence -> source

answerable as a query rather than a reconstruction from logs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..config import get_config
from ..obs import current_attribution, get_logger
from .store import Store, get_store, utcnow


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "model_dump"):        # pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):              # pydantic v1
        return obj.dict()
    return obj


class ArtifactStore:
    def __init__(self, store: Store | None = None, root: Path | None = None) -> None:
        cfg = get_config()
        self.store = store or get_store()
        self.root = root or cfg.artifacts_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = get_logger()

    def _dir_for(self, pipeline_id: str | None) -> Path:
        d = self.root / (pipeline_id or "unscoped")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(
        self,
        kind: str,
        payload: Any,
        *,
        pipeline_id: str | None = None,
        parent: str | None = None,
        meta: dict[str, Any] | None = None,
        extension: str = "json",
    ) -> str:
        """Persist an artifact and return its id."""
        attribution = current_attribution()
        pipeline_id = pipeline_id or attribution.get("pipeline_id")
        data = _jsonable(payload)

        if extension == "json":
            blob = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        elif isinstance(data, bytes):
            blob = data
        else:
            blob = str(data).encode("utf-8")

        sha = hashlib.sha256(blob).hexdigest()
        directory = self._dir_for(pipeline_id)
        filename = f"{kind}_{sha[:10]}.{extension}"
        path = directory / filename
        path.write_bytes(blob)

        artifact_id = self.store.add_artifact(
            kind=kind,
            pipeline_id=pipeline_id,
            path=str(path.relative_to(get_config().root)),
            sha256=sha,
            parent_artifact_id=parent,
            meta={**(meta or {}), "agent": attribution.get("agent"),
                  "skill": attribution.get("skill"), "state": attribution.get("state")},
            created_by=attribution.get("agent"),
        )
        self.log.debug(f"artifact {kind} written", artifact=artifact_id)
        return artifact_id

    def read(self, artifact_id: str) -> Any:
        row = self.store.query_one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
        if not row or not row["path"]:
            return None
        path = get_config().root / row["path"]
        if not path.exists():
            return None
        raw = path.read_bytes()
        if path.suffix == ".json":
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def latest(self, pipeline_id: str, kind: str) -> Any:
        rows = self.store.artifacts_for(pipeline_id, kind)
        return self.read(rows[-1]["id"]) if rows else None

    def latest_id(self, pipeline_id: str, kind: str) -> str | None:
        rows = self.store.artifacts_for(pipeline_id, kind)
        return rows[-1]["id"] if rows else None

    # -- provenance --------------------------------------------------------
    def provenance(self, artifact_id: str) -> list[dict[str, Any]]:
        """Full chain from an artifact back to its root, oldest last."""
        chain = self.store.artifact_chain(artifact_id)
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "path": r["path"],
                "sha256": r["sha256"],
                "created_at": r["created_at"],
                "created_by": r["created_by"],
                "meta": json.loads(r["meta_json"] or "{}"),
            }
            for r in chain
        ]

    def provenance_report(self, pipeline_id: str) -> dict[str, Any]:
        """Human-readable trail for one manuscript change.

        This is what the email body, the PR description and the changelog entry are
        all rendered from, so the same evidence trail appears everywhere.
        """
        pipe = self.store.query_one("SELECT * FROM pipelines WHERE id=?", (pipeline_id,))
        if not pipe:
            return {}

        artifacts = self.store.artifacts_for(pipeline_id)
        cluster = self.store.query_one(
            "SELECT * FROM clusters WHERE id=?", (pipe["cluster_id"],)
        ) if pipe["cluster_id"] else None
        verdict = self.store.query_one(
            "SELECT * FROM verdicts WHERE id=?", (pipe["verdict_id"],)
        ) if pipe["verdict_id"] else None

        sources: list[dict[str, Any]] = []
        if cluster:
            rows = self.store.query(
                "SELECT ri.*, cm.role FROM research_items ri "
                "JOIN cluster_members cm ON cm.research_item_id = ri.id "
                "WHERE cm.cluster_id = ? ORDER BY ri.published_at DESC",
                (cluster["id"],),
            )
            for r in rows:
                sources.append({
                    "source": r["source"], "title": r["title"], "url": r["url"],
                    "published_at": r["published_at"], "role": r["role"],
                    "lifecycle": r["lifecycle"],
                })

        evidence: list[dict[str, Any]] = []
        if cluster:
            rows = self.store.query(
                "SELECT e.* FROM evidence e "
                "JOIN cluster_members cm ON cm.research_item_id = e.research_item_id "
                "WHERE cm.cluster_id = ?",
                (cluster["id"],),
            )
            evidence = [
                {"kind": r["kind"], "strength": r["strength"], "statement": r["statement"]}
                for r in rows
            ]

        transitions = self.store.query(
            "SELECT from_state, to_state, at, note FROM pipeline_transitions "
            "WHERE pipeline_id=? ORDER BY id", (pipeline_id,),
        )

        return {
            "pipeline_id": pipeline_id,
            "state": pipe["state"],
            "created_at": pipe["created_at"],
            "updated_at": pipe["updated_at"],
            "cluster": {
                "id": cluster["id"], "title": cluster["title"],
                "maturity": cluster["maturity"],
                "concepts": json.loads(cluster["concepts_json"] or "[]"),
            } if cluster else None,
            "verdict": {
                "decision": verdict["decision"], "rationale": verdict["rationale"],
                "scope": verdict["scope"], "confidence": verdict["confidence"],
            } if verdict else None,
            "sources": sources,
            "evidence": evidence,
            "artifacts": [
                {"kind": r["kind"], "id": r["id"], "path": r["path"],
                 "created_by": r["created_by"], "created_at": r["created_at"]}
                for r in artifacts
            ],
            "transitions": [
                {"from": r["from_state"], "to": r["to_state"], "at": r["at"], "note": r["note"]}
                for r in transitions
            ],
        }


_artifacts: ArtifactStore | None = None


def get_artifacts() -> ArtifactStore:
    global _artifacts
    if _artifacts is None:
        _artifacts = ArtifactStore()
    return _artifacts
