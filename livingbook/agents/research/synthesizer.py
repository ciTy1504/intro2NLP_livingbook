"""Research Synthesizer.

Has no SEARCH and no FETCH capability by contract. It reads what the source agents
already persisted and produces clusters — which is what keeps every piece of evidence
traceable to the agent and source that collected it.
"""

from __future__ import annotations

import json
from typing import Any

from ...config import get_config
from ...research.models import ResearchCluster
from ...state.store import utcnow
from ..base import BaseAgent


class ResearchSynthesizer(BaseAgent[list[ResearchCluster]]):
    name = "research_synthesizer"
    uses_skills = ("research_synthesis",)

    def degraded_result(self) -> list[ResearchCluster]:
        return []

    async def execute(
        self, *, lifecycles: tuple[str, ...] = ("DISCOVERED", "SEEN", "MONITOR"),
        limit: int = 200, max_clusters: int | None = None, **_: Any,
    ) -> list[ResearchCluster]:
        items = self._load_undigested(lifecycles, limit)
        if not items:
            self.log.info("synthesizer: nothing undigested in research memory")
            return []

        self.log.info(f"synthesizer: {len(items)} items to synthesise")
        clusters = await self.skill("research_synthesis")(
            self.ctx, items=items, max_clusters=max_clusters)

        await self._persist(clusters)
        self.log.info(
            f"synthesizer: {len(clusters)} clusters "
            f"({', '.join(f'{c.maturity.value}:{c.title[:28]}' for c in clusters[:5])})",
            status="ok")
        return clusters

    def _load_undigested(self, lifecycles: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
        """Items not yet part of a cluster, plus MONITOR items due for re-evaluation.

        Re-examining MONITOR items is what turns this from a discovery system into one
        that tracks maturity: a technique that was speculative in March may have an
        implementation and a replication by June, and only a re-evaluation notices.
        """
        cfg = get_config()
        placeholders = ",".join("?" * len(lifecycles))
        rows = self.store.query(
            f"""
            SELECT ri.* FROM research_items ri
            WHERE ri.lifecycle IN ({placeholders})
              AND (
                    ri.id NOT IN (SELECT research_item_id FROM cluster_members)
                 OR (ri.lifecycle = 'MONITOR'
                     AND julianday('now') - julianday(ri.lifecycle_at) >= ?)
              )
            ORDER BY ri.discovered_at DESC
            LIMIT ?
            """,
            (*lifecycles, int(cfg.get("verdict.monitor_reevaluation_days", 7)), limit),
        )
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.get("payload_json") or "{}")
            except json.JSONDecodeError:
                d["payload"] = {}
            try:
                d["concepts"] = json.loads(d.get("concepts_json") or "[]")
            except json.JSONDecodeError:
                d["concepts"] = []
            out.append(d)
        return out

    async def _persist(self, clusters: list[ResearchCluster]) -> None:
        if not clusters:
            return
        records = []
        for c in clusters:
            members = (
                [{"research_item_id": i, "role": "paper"} for i in c.papers]
                + [{"research_item_id": i, "role": "implementation"}
                   for i in c.implementations]
                + [{"research_item_id": i, "role": "benchmark"} for i in c.benchmarks]
                + [{"research_item_id": i, "role": "community"}
                   for i in c.community_signals]
                + [{"research_item_id": i, "role": "blog"} for i in c.blogs]
            )
            records.append({
                "id": c.id, "title": c.title, "concepts": c.concepts,
                "maturity": c.maturity.value, "state": "SYNTHESIZED",
                "members": [m for m in members if m["research_item_id"]],
                "payload": c.model_dump(mode="json"),
            })
        await self.ctx.call("kb_upsert", kind="cluster", records=records)

        # Link cluster to concept in the graph so that later work can navigate
        # research -> concept -> book section without re-deriving it.
        edges = []
        for c in clusters:
            for concept in c.concepts:
                edges.append({
                    "src_type": "cluster", "src_id": c.id, "rel": "mentions",
                    "dst_type": "concept_name", "dst_id": concept.lower(),
                    "weight": 1.0,
                })
            for member in c.member_ids():
                edges.append({
                    "src_type": "cluster", "src_id": c.id, "rel": "derived_from",
                    "dst_type": "research_item", "dst_id": member, "weight": 1.0,
                })
        if edges:
            await self.ctx.call("kb_upsert", kind="graph_edge", records=edges)
