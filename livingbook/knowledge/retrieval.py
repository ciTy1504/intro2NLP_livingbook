"""Hierarchical retrieval over the book.

The rule from the architecture: never put the whole manuscript in a context. This
module is how that rule is kept. It offers three tiers, and callers pick the cheapest
one that answers their question:

  outline()        the TOC + chapter summaries — a few KB, covers the whole book
  retrieve()       the N sections most relevant to a query, summaries only
  context_for()    full LaTeX source for specific sections, under a token budget

Ranking is hybrid: concept-graph hits (precise, cheap) fused with embedding similarity
(recall for phrasings the concept index missed), plus a lexical fallback so the system
still functions before the semantic index has been built.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import get_config
from ..llm import get_provider
from ..state.store import Store, get_store
from .graph import KnowledgeGraph
from .latex import strip_latex

# ~4 chars per token for Latin text; Vietnamese with diacritics runs denser, so this
# deliberately under-estimates capacity rather than overflowing a context.
CHARS_PER_TOKEN = 3.2


@dataclass
class RetrievedNode:
    node_id: str
    kind: str
    number: str
    title: str
    file: str
    start_line: int
    end_line: int
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    score: float = 0.0
    matched_by: list[str] = field(default_factory=list)
    body: str | None = None

    @property
    def ref(self) -> str:
        num = f" {self.number}" if self.number else ""
        return f"{self.kind}{num} — {self.title}"

    def as_dict(self, include_body: bool = False) -> dict[str, Any]:
        out = {
            "node_id": self.node_id, "ref": self.ref, "file": self.file,
            "lines": [self.start_line, self.end_line], "summary": self.summary,
            "key_points": self.key_points, "concepts": self.concepts,
            "citations": self.citations, "score": round(self.score, 3),
            "matched_by": self.matched_by,
        }
        if include_body and self.body:
            out["body"] = self.body
        return out


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class BookRetriever:
    def __init__(self, store: Store | None = None) -> None:
        self.cfg = get_config()
        self.store = store or get_store()
        self.graph = KnowledgeGraph(self.store)
        self.manuscript: Path = self.cfg.manuscript_dir

    # -- tier 1: outline ---------------------------------------------------
    def outline(self, *, with_summaries: bool = True, max_depth: str = "section") -> str:
        """A whole-book map small enough to include in any prompt.

        This is what the global QA tier and the placement skill reason over instead of
        the manuscript itself.
        """
        depth_rank = {"part": 0, "chapter": 1, "section": 2, "subsection": 3,
                      "subsubsection": 4}
        cutoff = depth_rank.get(max_depth, 2)
        rows = self.store.query(
            "SELECT bn.*, ns.summary FROM book_nodes bn "
            "LEFT JOIN node_summaries ns ON ns.node_id = bn.id ORDER BY bn.order_idx")
        lines: list[str] = []
        for r in rows:
            d = depth_rank.get(r["kind"], 9)
            if d > cutoff:
                continue
            indent = "  " * d
            num = f"{r['number']} " if r["number"] else ""
            lines.append(f"{indent}{num}{r['title']}  [{r['id']}]")
            if with_summaries and r["summary"] and d <= 2:
                lines.append(f"{indent}   {r['summary']}")
        return "\n".join(lines)

    def table_of_contents(self) -> list[dict[str, Any]]:
        rows = self.store.query(
            "SELECT id, kind, number, title, file, start_line, end_line, word_count "
            "FROM book_nodes ORDER BY order_idx")
        return [dict(r) for r in rows]

    # -- tier 2: ranked retrieval -----------------------------------------
    async def retrieve(
        self,
        query: str,
        *,
        concepts: Iterable[str] | None = None,
        limit: int = 8,
        kinds: Iterable[str] = ("section", "subsection", "subsubsection"),
    ) -> list[RetrievedNode]:
        """Rank sections by relevance using concepts + embeddings + lexical overlap."""
        kinds = tuple(kinds)
        scores: dict[str, float] = {}
        matched_by: dict[str, set[str]] = {}

        def bump(node_id: str, amount: float, why: str) -> None:
            scores[node_id] = scores.get(node_id, 0.0) + amount
            matched_by.setdefault(node_id, set()).add(why)

        # (a) concept graph — precise, and free
        concept_terms = list(concepts or []) or _keywords(query)
        for node_id, score in self.graph.nodes_for_concepts(
            self.graph.find_concepts(concept_terms).values()
        ).items():
            bump(node_id, score * 1.0, "concept")

        # (b) embeddings — recall for phrasings the concept index missed
        stored = self.store.all_embeddings("node")
        if stored:
            try:
                resp = await get_provider().embed([query])
                if resp.vectors and resp.vectors[0]:
                    qvec = resp.vectors[0]
                    sims = sorted(
                        ((nid, cosine(qvec, vec)) for nid, vec in stored),
                        key=lambda kv: -kv[1],
                    )[: limit * 3]
                    for node_id, sim in sims:
                        if sim > 0.35:
                            bump(node_id, sim * 1.4, "embedding")
            except Exception:
                pass  # degrade to concept + lexical rather than fail retrieval

        # (c) lexical — the only tier that works before semantic indexing exists
        for node_id, score in self._lexical_scores(query, limit * 3).items():
            bump(node_id, score * 0.6, "lexical")

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: list[RetrievedNode] = []
        for node_id, score in ranked:
            node = self._load(node_id)
            if not node or node.kind not in kinds:
                continue
            node.score = score
            node.matched_by = sorted(matched_by.get(node_id, ()))
            out.append(node)
            if len(out) >= limit:
                break
        return out

    def _lexical_scores(self, query: str, limit: int) -> dict[str, float]:
        terms = _keywords(query)
        if not terms:
            return {}
        scores: dict[str, float] = {}
        like = " OR ".join(["bn.title LIKE ?"] * len(terms) +
                           ["ns.summary LIKE ?"] * len(terms))
        params = [f"%{t}%" for t in terms] * 2
        rows = self.store.query(
            f"SELECT bn.id, bn.title, ns.summary FROM book_nodes bn "
            f"LEFT JOIN node_summaries ns ON ns.node_id = bn.id WHERE {like} LIMIT ?",
            (*params, limit),
        )
        for r in rows:
            haystack = f"{r['title']} {r['summary'] or ''}".lower()
            hits = sum(1 for t in terms if t in haystack)
            if hits:
                scores[r["id"]] = hits / len(terms)
        return scores

    # -- tier 3: full source ----------------------------------------------
    def context_for(
        self,
        node_ids: Iterable[str],
        *,
        max_tokens: int = 24000,
        include_parents: bool = True,
        plain_text: bool = False,
    ) -> dict[str, Any]:
        """Actual manuscript source for specific nodes, truncated to a token budget.

        Returns what was included *and what was dropped*, so a caller can tell the
        difference between "the book does not say that" and "we ran out of budget".
        """
        budget = int(max_tokens * CHARS_PER_TOKEN)
        used = 0
        included: list[dict[str, Any]] = []
        dropped: list[str] = []
        seen: set[str] = set()

        ordered: list[str] = []
        for node_id in node_ids:
            if include_parents:
                for ancestor in reversed(self._ancestors(node_id)):
                    if ancestor not in seen:
                        seen.add(ancestor)
                        ordered.append(ancestor)
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)

        for node_id in ordered:
            node = self._load(node_id, with_body=True)
            if not node or node.body is None:
                continue
            body = strip_latex(node.body) if plain_text else node.body
            cost = len(body)
            # Ancestors are included as a heading line only — their prose is context
            # the caller did not ask for and would crowd out the real target.
            is_ancestor = node.kind in ("part", "chapter") and node_id not in set(node_ids)
            if is_ancestor:
                body = f"% [{node.ref}]"
                cost = len(body)
            if used + cost > budget:
                dropped.append(node.ref)
                continue
            used += cost
            included.append({
                "node_id": node.node_id, "ref": node.ref, "file": node.file,
                "lines": [node.start_line, node.end_line], "content": body,
            })

        return {
            "included": included,
            "dropped": dropped,
            "approx_tokens": int(used / CHARS_PER_TOKEN),
            "budget_tokens": max_tokens,
        }

    def render_context(self, ctx: dict[str, Any]) -> str:
        """Format a context bundle for a prompt."""
        parts = []
        for item in ctx["included"]:
            parts.append(
                f"--- {item['ref']}  ({item['file']}:{item['lines'][0]}-{item['lines'][1]}) "
                f"[{item['node_id']}] ---\n{item['content']}"
            )
        text = "\n\n".join(parts)
        if ctx.get("dropped"):
            text += ("\n\n[omitted for context budget: "
                     + ", ".join(ctx["dropped"][:8]) + "]")
        return text

    # -- loading -----------------------------------------------------------
    def _load(self, node_id: str, *, with_body: bool = False) -> RetrievedNode | None:
        row = self.store.query_one(
            "SELECT bn.*, ns.summary, ns.key_points_json FROM book_nodes bn "
            "LEFT JOIN node_summaries ns ON ns.node_id = bn.id WHERE bn.id = ?", (node_id,))
        if not row:
            return None
        node = RetrievedNode(
            node_id=row["id"], kind=row["kind"], number=row["number"] or "",
            title=row["title"], file=row["file"],
            start_line=row["start_line"], end_line=row["end_line"],
            summary=row["summary"] or "",
            key_points=json.loads(row["key_points_json"] or "[]"),
        )
        node.concepts = [
            r["canonical_name"] for r in self.store.query(
                "SELECT c.canonical_name FROM node_concepts nc "
                "JOIN concepts c ON c.id = nc.concept_id WHERE nc.node_id = ?", (node_id,))
        ]
        node.citations = [
            r["bib_key"] for r in self.store.query(
                "SELECT bib_key FROM cite_edges WHERE node_id = ?", (node_id,))
        ]
        if with_body:
            node.body = self.read_source(row["file"], row["start_line"], row["end_line"])
        return node

    def read_source(self, rel_file: str, start_line: int, end_line: int) -> str:
        path = self.manuscript / rel_file
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[max(0, start_line - 1):end_line])

    def _ancestors(self, node_id: str) -> list[str]:
        out: list[str] = []
        current = node_id
        seen: set[str] = set()
        while True:
            row = self.store.query_one(
                "SELECT parent_id FROM book_nodes WHERE id = ?", (current,))
            if not row or not row["parent_id"] or row["parent_id"] in seen:
                break
            seen.add(row["parent_id"])
            out.append(row["parent_id"])
            current = row["parent_id"]
        return out

    # -- convenience for agents -------------------------------------------
    def node_by_label(self, label: str) -> RetrievedNode | None:
        row = self.store.query_one("SELECT id FROM book_nodes WHERE label = ?", (label,))
        return self._load(row["id"]) if row else None

    def siblings(self, node_id: str) -> list[RetrievedNode]:
        row = self.store.query_one(
            "SELECT parent_id FROM book_nodes WHERE id = ?", (node_id,))
        if not row or not row["parent_id"]:
            return []
        rows = self.store.query(
            "SELECT id FROM book_nodes WHERE parent_id = ? ORDER BY order_idx",
            (row["parent_id"],))
        return [n for n in (self._load(r["id"]) for r in rows) if n]

    def neighbours_in_reading_order(self, node_id: str, window: int = 1) -> list[RetrievedNode]:
        row = self.store.query_one(
            "SELECT order_idx, kind FROM book_nodes WHERE id = ?", (node_id,))
        if not row:
            return []
        rows = self.store.query(
            "SELECT id FROM book_nodes WHERE kind = ? AND order_idx BETWEEN ? AND ? "
            "ORDER BY order_idx",
            (row["kind"], row["order_idx"] - window, row["order_idx"] + window))
        return [n for n in (self._load(r["id"]) for r in rows) if n and n.node_id != node_id]


_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "with", "is", "are",
    "how", "what", "why", "does", "do", "using", "use", "new", "that", "this", "it",
}


def _keywords(query: str, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out
