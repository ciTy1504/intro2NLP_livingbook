"""Builds and refreshes the Book Knowledge Base.

The indexer is what makes the rest of the system affordable. It converts the
manuscript into an addressable, summarised, embedded structure so that later stages
retrieve a few hundred relevant lines instead of loading a 143k-word book.

It is incremental by content hash: a re-index after a one-section change re-summarises
one section and leaves the other several hundred nodes untouched, which matters
because the LLM work here (summaries, concepts, claims, embeddings) is the single
largest quota consumer in the system.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import get_config
from ..llm import GenerationOptions, get_provider
from ..obs import get_logger, skill_scope
from ..state.store import Store, get_store, utcnow
from .bib import Bibliography
from .latex import LatexParser, ParsedBook, TexNode, collect_labels, strip_latex

# Nodes below this size are summarised together with their parent rather than on
# their own: a 40-word subsubsection has no summary worth an API call.
MIN_WORDS_FOR_SUMMARY = 120

SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING",
                    "description": "2-4 sentences, English, what this section teaches"},
        "key_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "concepts": {
            "type": "ARRAY",
            "description": "Technical concepts explained here, English canonical names",
            "items": {"type": "STRING"},
        },
        "prerequisites": {"type": "ARRAY", "items": {"type": "STRING"}},
        "level": {"type": "STRING", "enum": ["introductory", "intermediate", "advanced"]},
    },
    "required": ["summary", "key_points", "concepts", "level"],
}

CLAIMS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING",
                             "description": "The claim in English, self-contained"},
                    "claim_type": {
                        "type": "STRING",
                        "enum": ["numerical", "benchmark", "historical", "causal", "sota",
                                 "definitional", "architectural", "attribution"],
                    },
                    "needs_citation": {"type": "BOOLEAN"},
                    "existing_citation": {"type": "STRING"},
                },
                "required": ["text", "claim_type", "needs_citation"],
            },
        }
    },
    "required": ["claims"],
}


@dataclass
class IndexStats:
    files: int = 0
    nodes: int = 0
    nodes_new: int = 0
    nodes_changed: int = 0
    nodes_unchanged: int = 0
    summaries: int = 0
    concepts: int = 0
    claims: int = 0
    embeddings: int = 0
    figures: int = 0
    bib_entries: int = 0
    cite_edges: int = 0
    xrefs: int = 0
    graph_edges: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class BookIndexer:
    def __init__(self, store: Store | None = None) -> None:
        self.cfg = get_config()
        self.store = store or get_store()
        self.log = get_logger()
        self.manuscript = self.cfg.manuscript_dir
        self.parser = LatexParser(self.manuscript, self.cfg.get("project.main_tex", "main.tex"))

    # -- structural pass (no LLM) -----------------------------------------
    def index_structure(self) -> tuple[ParsedBook, IndexStats]:
        """Parse the manuscript and persist structure, citations, xrefs and figures."""
        stats = IndexStats()
        book = self.parser.parse()
        stats.files = len(book.files)
        stats.nodes = len(book.nodes)

        existing = {
            r["id"]: r["content_sha256"]
            for r in self.store.query("SELECT id, content_sha256 FROM book_nodes")
        }
        current_ids = {n.id for n in book.nodes}

        with self.store.transaction() as conn:
            for node in book.nodes:
                prior = existing.get(node.id)
                if prior is None:
                    stats.nodes_new += 1
                elif prior != node.content_sha256:
                    stats.nodes_changed += 1
                else:
                    stats.nodes_unchanged += 1

                # UPSERT, not INSERT OR REPLACE. REPLACE is implemented as DELETE +
                # INSERT, and node_summaries/claims/node_concepts carry
                # ON DELETE CASCADE onto book_nodes — so a plain REPLACE would silently
                # wipe the entire semantic index on every structural re-index, making
                # the content-hash incrementality below pointless.
                conn.execute(
                    "INSERT INTO book_nodes (id, kind, number, title, label, file, "
                    "start_line, end_line, parent_id, order_idx, word_count, content_sha256) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "kind=excluded.kind, number=excluded.number, title=excluded.title, "
                    "label=excluded.label, file=excluded.file, "
                    "start_line=excluded.start_line, end_line=excluded.end_line, "
                    "parent_id=excluded.parent_id, order_idx=excluded.order_idx, "
                    "word_count=excluded.word_count, content_sha256=excluded.content_sha256",
                    (node.id, node.kind, node.number, node.title, node.label, node.file,
                     node.start_line, node.end_line, node.parent_id, node.order_idx,
                     node.word_count, node.content_sha256),
                )

            # Drop nodes that no longer exist (a section was deleted or renamed).
            stale = set(existing) - current_ids
            for node_id in stale:
                conn.execute("DELETE FROM book_nodes WHERE id = ?", (node_id,))

            conn.execute("DELETE FROM cite_edges")
            conn.execute("DELETE FROM xrefs")
            for node in book.nodes:
                counts: dict[str, int] = {}
                for key in node.cites:
                    counts[key] = counts.get(key, 0) + 1
                for key, n in counts.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO cite_edges (node_id, bib_key, count) VALUES (?,?,?)",
                        (node.id, key, n))
                    stats.cite_edges += 1
                for kind, label in node.refs:
                    conn.execute(
                        "INSERT OR REPLACE INTO xrefs (src_node_id, label, dst_node_id, kind) "
                        "VALUES (?,?,?,?)", (node.id, label, None, kind))
                    stats.xrefs += 1

        self._resolve_xrefs(book)
        stats.figures = self._index_figures(book)
        stats.bib_entries = self._index_bib()
        stats.graph_edges = self._build_structural_graph(book)

        self.log.info(
            f"structure indexed: {stats.nodes} nodes "
            f"({stats.nodes_new} new, {stats.nodes_changed} changed, "
            f"{stats.nodes_unchanged} unchanged), {stats.cite_edges} citations",
            status="ok",
        )
        return book, stats

    def _resolve_xrefs(self, book: ParsedBook) -> None:
        """Point each \\ref at the node that owns the label, where one exists."""
        label_owner: dict[str, str] = {}
        for node in book.nodes:
            if node.label:
                label_owner[node.label] = node.id
        # Labels belonging to figures, equations and tcolorbox theorem environments
        # resolve to the node whose body contains them. collect_labels is used rather
        # than a bare \label scan because a theorem's label is its environment prefix
        # plus the key at the call site, which a plain scan never sees.
        for node in book.nodes:
            for label in collect_labels(node.body):
                label_owner.setdefault(label, node.id)

        with self.store.transaction() as conn:
            for label, owner in label_owner.items():
                conn.execute("UPDATE xrefs SET dst_node_id = ? WHERE label = ?", (owner, label))

    def _index_figures(self, book: ParsedBook) -> int:
        images_dir = self.cfg.images_dir
        n = 0
        with self.store.transaction() as conn:
            for node in book.nodes:
                for fig in node.figures:
                    path, sha, status = "", None, "missing"
                    for ext in (".png", ".jpg", ".jpeg", ".pdf"):
                        candidate = images_dir / f"{fig.key}{ext}"
                        if candidate.exists():
                            path = f"images/{fig.key}{ext}"
                            sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
                            status = "placed"
                            break
                    if fig.kind == "includegraphics" and fig.path:
                        direct = self.manuscript / fig.path
                        if direct.exists():
                            path = fig.path
                            sha = hashlib.sha256(direct.read_bytes()).hexdigest()
                            status = "placed"
                    # UPSERT that refreshes what the manuscript owns (location,
                    # requirement, caption, file presence) while preserving what the
                    # Visual Engine owns (licence, attribution, source, alt text).
                    conn.execute(
                        "INSERT INTO figures (key, node_id, path, requirement, caption, "
                        "status, sha256, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "node_id=excluded.node_id, path=excluded.path, "
                        "requirement=excluded.requirement, caption=excluded.caption, "
                        "status=excluded.status, sha256=excluded.sha256, "
                        "updated_at=excluded.updated_at",
                        (fig.key, node.id, path, fig.requirement, fig.caption, status, sha,
                         utcnow()),
                    )
                    n += 1
        return n

    def _index_bib(self) -> int:
        bib = Bibliography(self.cfg.bib_path)
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM bib_entries")
            for entry in bib.entries.values():
                row = entry.to_row()
                conn.execute(
                    "INSERT OR REPLACE INTO bib_entries (bib_key, entry_type, title, authors, "
                    "year, venue, doi, arxiv_id, url, raw) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (row["bib_key"], row["entry_type"], row["title"], row["authors"],
                     row["year"], row["venue"], row["doi"], row["arxiv_id"],
                     row["url"], row["raw"]),
                )
        return len(bib)

    def _build_structural_graph(self, book: ParsedBook) -> int:
        edges: list[tuple] = []
        for node in book.nodes:
            if node.parent_id:
                edges.append(("node", node.parent_id, "contains", "node", node.id, 1.0, None))
            for key in set(node.cites):
                edges.append(("node", node.id, "cited_by", "bib", key, 1.0, None))
            for fig in node.figures:
                edges.append(("node", node.id, "illustrated_by", "figure", fig.key, 1.0, None))
        self.store.execute(
            "DELETE FROM graph_edges WHERE rel IN ('contains','cited_by','illustrated_by')"
        )
        self.store.add_edges(edges)
        return len(edges)

    # -- semantic pass (LLM) ----------------------------------------------
    async def index_semantics(
        self, book: ParsedBook | None = None, *, force: bool = False,
        concurrency: int = 6, limit: int | None = None,
    ) -> IndexStats:
        """Summarise nodes, extract concepts and claims, and embed everything.

        Skips any node whose content hash already has a summary, so a re-index after
        a small edit costs a handful of calls rather than several hundred.
        """
        stats = IndexStats()
        book = book or self.parser.parse()
        provider = get_provider()

        targets = [
            n for n in book.nodes
            if n.kind in ("section", "subsection", "subsubsection", "chapter")
            and n.word_count >= MIN_WORDS_FOR_SUMMARY
        ]
        if not force:
            done = {
                r["node_id"]: r["source_sha256"]
                for r in self.store.query("SELECT node_id, source_sha256 FROM node_summaries")
            }
            targets = [n for n in targets if done.get(n.id) != n.content_sha256]
        if limit:
            targets = targets[:limit]

        if not targets:
            self.log.info("semantic index already current; nothing to do")
            return stats

        self.log.info(f"summarising {len(targets)} nodes (concurrency {concurrency})")
        sem = asyncio.Semaphore(concurrency)
        concept_names: set[str] = set()

        async def process(node: TexNode) -> tuple[TexNode, dict[str, Any] | None]:
            async with sem:
                try:
                    return node, await self._summarise_node(provider, node, book)
                except Exception as exc:
                    self.log.warn(f"summary failed for {node.id}: {type(exc).__name__}: {exc}")
                    return node, None

        # Persist as each node completes rather than gathering everything first.
        # Indexing the whole book is hundreds of calls over many minutes; batching the
        # writes to the end means an interruption loses all of it, and leaves no way to
        # watch progress.
        with skill_scope("book_indexing"):
            tasks = [asyncio.create_task(process(n)) for n in targets]
            done_count = 0
            for future in asyncio.as_completed(tasks):
                node, result = await future
                done_count += 1
                if result:
                    stats.summaries += 1
                    self._persist_summary(node, result)
                    concept_names.update(result.get("concepts", []) or [])
                    stats.claims += len(result.get("claims", []) or [])
                if done_count % 25 == 0 or done_count == len(targets):
                    self.log.info(
                        f"summarised {done_count}/{len(targets)} nodes "
                        f"({stats.summaries} ok, {stats.claims} claims)"
                    )

        stats.concepts = self._persist_concepts(concept_names)
        stats.embeddings = await self._embed_all(provider, book)
        self._build_semantic_graph()

        self.log.info(
            f"semantic index: {stats.summaries} summaries, {stats.concepts} concepts, "
            f"{stats.claims} claims, {stats.embeddings} embeddings", status="ok",
        )
        return stats

    async def _summarise_node(
        self, provider: Any, node: TexNode, book: ParsedBook,
    ) -> dict[str, Any]:
        text = strip_latex(node.body)[:14000]
        parent = book.by_id().get(node.parent_id) if node.parent_id else None
        context = f"Book: {self.cfg.get('project.book_title')} (a Vietnamese NLP textbook)\n"
        if parent:
            context += f"Parent: {parent.kind} {parent.number} — {parent.title}\n"

        summary_prompt = (
            f"{context}"
            f"Section: {node.kind} {node.number} — {node.title}\n\n"
            "Summarise this textbook section for a knowledge index. The section is in "
            "Vietnamese; answer in ENGLISH so the index is searchable in one language.\n"
            "Concept names must be the standard English technical terms "
            "(e.g. 'scaling laws', 'grouped-query attention'), not Vietnamese.\n\n"
            f"SECTION TEXT:\n{text}"
        )
        summary = await provider.generate_structured(
            summary_prompt, SUMMARY_SCHEMA, role="fast",
            options=GenerationOptions(temperature=0.1),
        )

        claims: list[dict[str, Any]] = []
        # Claim extraction is only worth its cost where claims actually live.
        if node.kind in ("section", "subsection") and node.word_count >= 200:
            claims_prompt = (
                f"{context}Section: {node.title}\n\n"
                "Extract the factual claims this section makes that a reader could "
                "check against a source. Answer in ENGLISH.\n\n"
                "Include: numerical results, benchmark scores, state-of-the-art "
                "assertions, historical/chronological statements, causal claims, and "
                "attributions of a method to its authors.\n"
                "Exclude: definitions of standard terms, pedagogical framing, worked "
                "examples, and statements about the book itself.\n"
                "Set needs_citation=true when the claim requires a source and the text "
                "shows none. Citations appear as [key] markers.\n\n"
                f"SECTION TEXT:\n{text}"
            )
            try:
                extracted = await provider.generate_structured(
                    claims_prompt, CLAIMS_SCHEMA, role="fast",
                    options=GenerationOptions(temperature=0.1),
                )
                claims = extracted.data.get("claims", []) or []
            except Exception as exc:
                self.log.debug(f"claim extraction skipped for {node.id}: {exc}")

        return {**summary.data, "claims": claims, "model": summary.model}

    def _persist_summary(self, node: TexNode, result: dict[str, Any]) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO node_summaries (node_id, summary, key_points_json, "
                "prerequisites_json, level, model, source_sha256, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (node.id, result.get("summary", ""),
                 json.dumps(result.get("key_points", []), ensure_ascii=False),
                 json.dumps(result.get("prerequisites", []), ensure_ascii=False),
                 result.get("level"), result.get("model"), node.content_sha256, utcnow()),
            )
            conn.execute("DELETE FROM claims WHERE node_id = ?", (node.id,))
            for i, claim in enumerate(result.get("claims", []) or []):
                cid = f"clm_{node.id}_{i}"
                conn.execute(
                    "INSERT OR REPLACE INTO claims (id, node_id, text, claim_type, "
                    "needs_citation, status, created_at, source_sha256) VALUES (?,?,?,?,?,?,?,?)",
                    (cid, node.id, claim.get("text", ""), claim.get("claim_type"),
                     1 if claim.get("needs_citation") else 0, "unverified",
                     utcnow(), node.content_sha256),
                )
                existing = (claim.get("existing_citation") or "").strip()
                for key in re.split(r"[,\s]+", existing):
                    key = key.strip("[]")
                    if key:
                        conn.execute(
                            "INSERT OR REPLACE INTO claim_citations "
                            "(claim_id, bib_key, support_status) VALUES (?,?,?)",
                            (cid, key, "unverified"),
                        )
            conn.execute("DELETE FROM node_concepts WHERE node_id = ?", (node.id,))
            for name in result.get("concepts", []) or []:
                canonical = str(name).strip()
                if not canonical:
                    continue
                cid = _concept_id(canonical)
                # The concept row must exist before the link row: node_concepts has a
                # foreign key onto concepts, and summaries are persisted per node
                # before _persist_concepts runs.
                conn.execute(
                    "INSERT OR IGNORE INTO concepts (id, canonical_name, aliases_json, "
                    "created_at) VALUES (?,?,?,?)",
                    (cid, canonical, "[]", utcnow()),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO node_concepts (node_id, concept_id, salience) "
                    "VALUES (?,?,?)", (node.id, cid, 0.8),
                )

    def _persist_concepts(self, names: Iterable[str]) -> int:
        """Backfill each concept's first appearance in reading order.

        The concept rows themselves are created in _persist_summary (the FK requires
        it); this pass only resolves `first_node_id`, which needs all links present.
        """
        n = 0
        with self.store.transaction() as conn:
            for name in {str(x).strip() for x in names if str(x).strip()}:
                cid = _concept_id(name)
                first = conn.execute(
                    "SELECT nc.node_id FROM node_concepts nc "
                    "JOIN book_nodes bn ON bn.id = nc.node_id "
                    "WHERE nc.concept_id = ? ORDER BY bn.order_idx LIMIT 1", (cid,)
                ).fetchone()
                conn.execute(
                    "UPDATE concepts SET first_node_id = ? WHERE id = ?",
                    (first["node_id"] if first else None, cid),
                )
                n += 1
        return n

    async def _embed_all(self, provider: Any, book: ParsedBook) -> int:
        """Embed node summaries and concept names for retrieval."""
        rows = self.store.query(
            "SELECT ns.node_id, ns.summary, bn.title FROM node_summaries ns "
            "JOIN book_nodes bn ON bn.id = ns.node_id"
        )
        pending_nodes = [
            (r["node_id"], f"{r['title']}. {r['summary']}")
            for r in rows if not self.store.has_embedding("node", r["node_id"])
        ]
        concepts = self.store.query("SELECT id, canonical_name FROM concepts")
        pending_concepts = [
            (c["id"], c["canonical_name"])
            for c in concepts if not self.store.has_embedding("concept", c["id"])
        ]

        total = 0
        for owner_type, pending in (("node", pending_nodes), ("concept", pending_concepts)):
            for start in range(0, len(pending), 64):
                chunk = pending[start:start + 64]
                if not chunk:
                    continue
                resp = await provider.embed([t for _, t in chunk])
                self.store.put_embeddings(
                    [(owner_type, oid, resp.model, vec)
                     for (oid, _), vec in zip(chunk, resp.vectors) if vec]
                )
                total += len(chunk)
        return total

    def _build_semantic_graph(self) -> int:
        edges: list[tuple] = []
        for r in self.store.query("SELECT node_id, concept_id, salience FROM node_concepts"):
            edges.append(("node", r["node_id"], "mentions", "concept", r["concept_id"],
                          r["salience"] or 0.5, None))
            edges.append(("concept", r["concept_id"], "explained_in", "node", r["node_id"],
                          r["salience"] or 0.5, None))
        for r in self.store.query("SELECT id, node_id, needs_citation FROM claims"):
            edges.append(("node", r["node_id"], "asserts", "claim", r["id"], 1.0, None))
            if r["needs_citation"]:
                edges.append(("claim", r["id"], "needs_evidence", "gap", r["id"], 1.0, None))
        for r in self.store.query("SELECT claim_id, bib_key FROM claim_citations"):
            edges.append(("claim", r["claim_id"], "cited_by", "bib", r["bib_key"], 1.0, None))
        for r in self.store.query(
            "SELECT src_node_id, dst_node_id FROM xrefs WHERE dst_node_id IS NOT NULL"
        ):
            edges.append(("node", r["src_node_id"], "related_to", "node", r["dst_node_id"],
                          0.5, None))

        self.store.execute(
            "DELETE FROM graph_edges WHERE rel IN "
            "('mentions','explained_in','asserts','needs_evidence','related_to')"
        )
        self.store.add_edges(edges)
        return len(edges)

    # -- convenience -------------------------------------------------------
    async def full_index(self, *, force: bool = False, limit: int | None = None) -> dict[str, Any]:
        book, structural = self.index_structure()
        semantic = await self.index_semantics(book, force=force, limit=limit)
        return {
            "structural": structural.as_dict(),
            "semantic": semantic.as_dict(),
            "book_stats": book.stats(),
        }

    def manuscript_fingerprint(self) -> str:
        """Hash of all manuscript sources — cheap check for 'did anything change'."""
        h = hashlib.sha256()
        for rel in sorted(self.parser.input_order()):
            path = self.manuscript / rel
            if path.exists():
                h.update(rel.encode())
                h.update(path.read_bytes())
        return h.hexdigest()


def _concept_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48]
    digest = hashlib.sha1(name.lower().encode()).hexdigest()[:8]
    return f"cpt_{slug}_{digest}" if slug else f"cpt_{digest}"
