"""Book knowledge graph queries.

The graph exists to answer one question well:

    given new research about concepts X, Y, Z — which parts of the book are affected?

Everything else it does is in service of that. Edges are persisted in SQLite
(`graph_edges`); networkx is used only for the algorithms that genuinely need a graph
library (connected components, shortest paths), not as the storage layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..state.store import Store, get_store


@dataclass
class ImpactHit:
    node_id: str
    title: str
    number: str
    kind: str
    file: str
    start_line: int
    end_line: int
    score: float
    reasons: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "title": self.title, "number": self.number,
            "kind": self.kind, "file": self.file,
            "lines": [self.start_line, self.end_line],
            "score": round(self.score, 3), "reasons": self.reasons,
            "concepts": self.concepts, "claims": self.claims, "citations": self.citations,
        }


class KnowledgeGraph:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or get_store()

    # -- concept lookup ----------------------------------------------------
    def find_concepts(self, names: Iterable[str]) -> dict[str, str]:
        """Map concept names to ids, matching canonical names and aliases loosely."""
        out: dict[str, str] = {}
        rows = self.store.query("SELECT id, canonical_name, aliases_json FROM concepts")
        lowered = {r["canonical_name"].lower(): r["id"] for r in rows}
        alias_map: dict[str, str] = {}
        for r in rows:
            for alias in json.loads(r["aliases_json"] or "[]"):
                alias_map[str(alias).lower()] = r["id"]

        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            if key in lowered:
                out[name] = lowered[key]
            elif key in alias_map:
                out[name] = alias_map[key]
            else:
                # Substring match both ways: "GQA" should reach "grouped-query
                # attention (GQA)" and "attention" should reach "attention mechanism".
                for canon, cid in lowered.items():
                    if key in canon or canon in key:
                        out[name] = cid
                        break
        return out

    def nodes_for_concepts(self, concept_ids: Iterable[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for cid in concept_ids:
            for edge in self.store.out_edges("concept", cid, "explained_in"):
                scores[edge["dst_id"]] = scores.get(edge["dst_id"], 0.0) + (edge["weight"] or 0.5)
        return scores

    # -- impact analysis ---------------------------------------------------
    def impact_of_concepts(
        self, concepts: Iterable[str], *, limit: int = 12, include_neighbours: bool = True,
    ) -> list[ImpactHit]:
        """Sections a research cluster would touch, ranked, with the reason for each.

        This is the query that keeps the book editable without reading it: the verdict
        agent gets back exactly the affected sections, their claims and their citations.
        """
        names = [c for c in concepts if c and c.strip()]
        matched = self.find_concepts(names)
        if not matched:
            return []

        scores = self.nodes_for_concepts(matched.values())

        if include_neighbours:
            # A section that cross-references an affected section is weakly affected.
            for node_id, base in list(scores.items()):
                for edge in self.store.out_edges("node", node_id, "related_to"):
                    scores[edge["dst_id"]] = scores.get(edge["dst_id"], 0.0) + base * 0.15
                for edge in self.store.in_edges("node", node_id, "contains"):
                    scores[edge["src_id"]] = scores.get(edge["src_id"], 0.0) + base * 0.10

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        hits: list[ImpactHit] = []
        reverse = {v: k for k, v in matched.items()}

        for node_id, score in ranked:
            row = self.store.query_one("SELECT * FROM book_nodes WHERE id = ?", (node_id,))
            if not row:
                continue
            node_concepts = [
                r["canonical_name"] for r in self.store.query(
                    "SELECT c.canonical_name FROM node_concepts nc "
                    "JOIN concepts c ON c.id = nc.concept_id WHERE nc.node_id = ?", (node_id,))
            ]
            claims = [
                r["text"] for r in self.store.query(
                    "SELECT text FROM claims WHERE node_id = ? LIMIT 12", (node_id,))
            ]
            citations = [
                r["bib_key"] for r in self.store.query(
                    "SELECT bib_key FROM cite_edges WHERE node_id = ?", (node_id,))
            ]
            overlapping = [
                reverse[cid] for cid in matched.values()
                if any(e["dst_id"] == node_id
                       for e in self.store.out_edges("concept", cid, "explained_in"))
            ]
            reasons = []
            if overlapping:
                reasons.append(f"explains {', '.join(overlapping[:4])}")
            if not overlapping:
                reasons.append("adjacent to an affected section")

            hits.append(ImpactHit(
                node_id=node_id, title=row["title"], number=row["number"] or "",
                kind=row["kind"], file=row["file"],
                start_line=row["start_line"], end_line=row["end_line"],
                score=score, reasons=reasons, concepts=node_concepts,
                claims=claims, citations=citations,
            ))
        return hits

    # -- traversal helpers -------------------------------------------------
    def claims_for_node(self, node_id: str) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM claims WHERE node_id = ?", (node_id,))
        out = []
        for r in rows:
            cites = [
                cc["bib_key"] for cc in self.store.query(
                    "SELECT bib_key FROM claim_citations WHERE claim_id = ?", (r["id"],))
            ]
            out.append({
                "id": r["id"], "text": r["text"], "type": r["claim_type"],
                "needs_citation": bool(r["needs_citation"]), "status": r["status"],
                "citations": cites,
            })
        return out

    def citation_neighbourhood(self, bib_key: str) -> dict[str, Any]:
        """Everywhere a source is used, and what it is claimed to support."""
        nodes = self.store.query(
            "SELECT bn.id, bn.title, bn.number, bn.kind, ce.count FROM cite_edges ce "
            "JOIN book_nodes bn ON bn.id = ce.node_id WHERE ce.bib_key = ?", (bib_key,))
        claims = self.store.query(
            "SELECT c.id, c.text, c.claim_type, cc.support_status FROM claim_citations cc "
            "JOIN claims c ON c.id = cc.claim_id WHERE cc.bib_key = ?", (bib_key,))
        entry = self.store.query_one(
            "SELECT * FROM bib_entries WHERE bib_key = ?", (bib_key,))
        return {
            "bib_key": bib_key,
            "entry": dict(entry) if entry else None,
            "cited_in": [dict(n) for n in nodes],
            "supports_claims": [dict(c) for c in claims],
        }

    def concept_coverage(self, concept_name: str) -> dict[str, Any]:
        matched = self.find_concepts([concept_name])
        if not matched:
            return {"concept": concept_name, "found": False, "nodes": []}
        cid = next(iter(matched.values()))
        rows = self.store.query(
            "SELECT bn.id, bn.title, bn.number, bn.kind, bn.file, nc.salience "
            "FROM node_concepts nc JOIN book_nodes bn ON bn.id = nc.node_id "
            "WHERE nc.concept_id = ? ORDER BY bn.order_idx", (cid,))
        return {
            "concept": concept_name, "found": True, "concept_id": cid,
            "nodes": [dict(r) for r in rows],
        }

    def duplicate_concept_coverage(self, min_nodes: int = 4) -> list[dict[str, Any]]:
        """Concepts explained in many separate places — candidate duplication."""
        rows = self.store.query(
            "SELECT c.canonical_name, c.id, COUNT(*) n FROM node_concepts nc "
            "JOIN concepts c ON c.id = nc.concept_id "
            "GROUP BY c.id HAVING n >= ? ORDER BY n DESC", (min_nodes,))
        return [{"concept": r["canonical_name"], "concept_id": r["id"], "nodes": r["n"]}
                for r in rows]

    # -- exports -----------------------------------------------------------
    def to_networkx(self) -> Any:
        import networkx as nx
        g = nx.MultiDiGraph()
        for r in self.store.query("SELECT * FROM graph_edges"):
            src = f"{r['src_type']}:{r['src_id']}"
            dst = f"{r['dst_type']}:{r['dst_id']}"
            g.add_edge(src, dst, key=r["rel"], weight=r["weight"] or 1.0)
        return g

    def stats(self) -> dict[str, Any]:
        rels = {
            r["rel"]: r["n"] for r in self.store.query(
                "SELECT rel, COUNT(*) n FROM graph_edges GROUP BY rel ORDER BY n DESC")
        }
        return {
            "edges": sum(rels.values()),
            "by_relation": rels,
            "concepts": int(self.store.scalar("SELECT COUNT(*) FROM concepts") or 0),
            "nodes": int(self.store.scalar("SELECT COUNT(*) FROM book_nodes") or 0),
            "claims": int(self.store.scalar("SELECT COUNT(*) FROM claims") or 0),
        }
