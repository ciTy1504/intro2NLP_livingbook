"""Knowledge-base tools — the agents' window onto the book and research memory.

``kb_retrieve_context`` is the one that enforces the architecture's central rule: it
returns manuscript source under an explicit token budget and reports what it dropped,
so no agent can accidentally pull the whole book into a prompt.
"""

from __future__ import annotations

import json
from typing import Any

from ..knowledge.graph import KnowledgeGraph
from ..knowledge.retrieval import BookRetriever
from ..state.store import get_store, utcnow
from .registry import Capability, ToolError, tool


@tool("kb_query", [Capability.KB_READ],
      description="Query the book knowledge base: outline, sections, claims, citations.")
async def kb_query(kind: str, **kwargs: Any) -> Any:
    store = get_store()
    retriever = BookRetriever(store)
    graph = KnowledgeGraph(store)

    if kind == "outline":
        return {"outline": retriever.outline(
            with_summaries=kwargs.get("with_summaries", True),
            max_depth=kwargs.get("max_depth", "section"))}

    if kind == "toc":
        return {"toc": retriever.table_of_contents()}

    if kind == "search":
        query = kwargs.get("query", "")
        hits = await retriever.retrieve(
            query, concepts=kwargs.get("concepts"),
            limit=int(kwargs.get("limit", 8)))
        return {"hits": [h.as_dict() for h in hits]}

    if kind == "node":
        node_id = kwargs.get("node_id") or ""
        node = retriever._load(node_id, with_body=kwargs.get("with_body", False))
        if not node:
            raise ToolError(f"no such node: {node_id}")
        return node.as_dict(include_body=kwargs.get("with_body", False))

    if kind == "node_by_label":
        node = retriever.node_by_label(kwargs.get("label", ""))
        return node.as_dict() if node else None

    if kind == "impact":
        concepts = kwargs.get("concepts") or []
        hits = graph.impact_of_concepts(concepts, limit=int(kwargs.get("limit", 12)))
        return {"affected": [h.as_dict() for h in hits]}

    if kind == "claims":
        node_id = kwargs.get("node_id")
        if node_id:
            return {"claims": graph.claims_for_node(node_id)}
        rows = store.query(
            "SELECT c.*, bn.title, bn.number, bn.file FROM claims c "
            "JOIN book_nodes bn ON bn.id = c.node_id "
            "WHERE (? = 0 OR c.needs_citation = 1) LIMIT ?",
            (1 if kwargs.get("needs_citation") else 0, int(kwargs.get("limit", 100))))
        return {"claims": [dict(r) for r in rows]}

    if kind == "citation":
        return graph.citation_neighbourhood(kwargs.get("bib_key", ""))

    if kind == "bib":
        key = kwargs.get("bib_key")
        if key:
            row = store.query_one("SELECT * FROM bib_entries WHERE bib_key = ?", (key,))
            return dict(row) if row else None
        rows = store.query("SELECT bib_key, title, year, venue, doi, arxiv_id "
                           "FROM bib_entries ORDER BY year DESC LIMIT ?",
                           (int(kwargs.get("limit", 50)),))
        return {"entries": [dict(r) for r in rows]}

    if kind == "concept":
        return graph.concept_coverage(kwargs.get("name", ""))

    if kind == "concepts":
        rows = store.query(
            "SELECT c.canonical_name, COUNT(nc.node_id) n FROM concepts c "
            "LEFT JOIN node_concepts nc ON nc.concept_id = c.id "
            "GROUP BY c.id ORDER BY n DESC LIMIT ?", (int(kwargs.get("limit", 200)),))
        return {"concepts": [{"name": r["canonical_name"], "sections": r["n"]} for r in rows]}

    if kind == "figures":
        status = kwargs.get("status")
        if status:
            rows = store.query("SELECT * FROM figures WHERE status = ?", (status,))
        else:
            rows = store.query("SELECT * FROM figures")
        return {"figures": [dict(r) for r in rows]}

    if kind == "research":
        lifecycle = kwargs.get("lifecycle")
        sql = "SELECT * FROM research_items"
        params: list[Any] = []
        if lifecycle:
            sql += " WHERE lifecycle = ?"
            params.append(lifecycle)
        sql += " ORDER BY discovered_at DESC LIMIT ?"
        params.append(int(kwargs.get("limit", 50)))
        return {"items": [_research_row(r) for r in store.query(sql, params)]}

    if kind == "clusters":
        rows = store.query(
            "SELECT * FROM clusters ORDER BY updated_at DESC LIMIT ?",
            (int(kwargs.get("limit", 25)),))
        return {"clusters": [dict(r) for r in rows]}

    if kind == "stats":
        return {"db": store.counts(), "graph": graph.stats()}

    raise ToolError(
        f"unknown kb_query kind {kind!r}; expected one of: outline, toc, search, node, "
        "node_by_label, impact, claims, citation, bib, concept, concepts, figures, "
        "research, clusters, stats"
    )


@tool("kb_retrieve_context", [Capability.KB_READ],
      description="Fetch manuscript source for specific nodes under a token budget.")
async def kb_retrieve_context(
    node_ids: list[str], *, max_tokens: int = 20000, plain_text: bool = False,
    include_parents: bool = True, rendered: bool = True,
) -> dict[str, Any]:
    retriever = BookRetriever()
    ctx = retriever.context_for(
        node_ids, max_tokens=max_tokens, plain_text=plain_text,
        include_parents=include_parents)
    if rendered:
        ctx["rendered"] = retriever.render_context(ctx)
    return ctx


@tool("kb_upsert", [Capability.KB_WRITE],
      description="Write research items, evidence, clusters or figure metadata.")
async def kb_upsert(kind: str, records: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    store = get_store()
    rows = records if isinstance(records, list) else [records]
    now = utcnow()
    written = 0

    if kind == "research_item":
        with store.transaction() as conn:
            for r in rows:
                existing = conn.execute(
                    "SELECT id, lifecycle FROM research_items WHERE source=? AND source_id=?",
                    (r["source"], r["source_id"])).fetchone()
                if existing:
                    # Re-discovery is not a new item: bump last_seen_at and promote
                    # DISCOVERED -> SEEN so the monitor sweep can tell a genuinely new
                    # result from one that keeps reappearing.
                    conn.execute(
                        "UPDATE research_items SET last_seen_at=?, payload_json=?, "
                        "lifecycle=CASE WHEN lifecycle='DISCOVERED' THEN 'SEEN' "
                        "ELSE lifecycle END WHERE id=?",
                        (now, json.dumps(r.get("payload", r), ensure_ascii=False, default=str),
                         existing["id"]))
                else:
                    conn.execute(
                        "INSERT INTO research_items (id, source, source_id, url, title, "
                        "summary, published_at, discovered_at, last_seen_at, lifecycle, "
                        "lifecycle_at, relevance, concepts_json, payload_json, content_sha256) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["id"], r["source"], r["source_id"], r.get("url", ""),
                         r.get("title", ""), r.get("summary", ""), r.get("published_at"),
                         now, now, r.get("lifecycle", "DISCOVERED"), now,
                         r.get("relevance"),
                         json.dumps(r.get("concepts", []), ensure_ascii=False),
                         json.dumps(r.get("payload", r), ensure_ascii=False, default=str),
                         r.get("content_sha256")))
                written += 1

    elif kind == "evidence":
        with store.transaction() as conn:
            for r in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO evidence (id, research_item_id, kind, statement, "
                    "strength, supports, provenance_json) VALUES (?,?,?,?,?,?,?)",
                    (r["id"], r["research_item_id"], r["kind"], r["statement"],
                     r.get("strength", "moderate"), r.get("supports"),
                     json.dumps(r.get("provenance", []), ensure_ascii=False, default=str)))
                written += 1

    elif kind == "cluster":
        with store.transaction() as conn:
            for r in rows:
                conn.execute(
                    "INSERT INTO clusters (id, title, concepts_json, maturity, state, "
                    "created_at, updated_at, payload_json) VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
                    "concepts_json=excluded.concepts_json, maturity=excluded.maturity, "
                    "state=excluded.state, updated_at=excluded.updated_at, "
                    "payload_json=excluded.payload_json",
                    (r["id"], r.get("title", ""),
                     json.dumps(r.get("concepts", []), ensure_ascii=False),
                     r.get("maturity"), r.get("state", "SYNTHESIZED"), now, now,
                     json.dumps(r.get("payload", r), ensure_ascii=False, default=str)))
                for member in r.get("members", []):
                    conn.execute(
                        "INSERT OR REPLACE INTO cluster_members (cluster_id, "
                        "research_item_id, role) VALUES (?,?,?)",
                        (r["id"], member["research_item_id"], member.get("role")))
                written += 1

    elif kind == "lifecycle":
        with store.transaction() as conn:
            for r in rows:
                conn.execute(
                    "UPDATE research_items SET lifecycle=?, lifecycle_at=?, "
                    "dismiss_reason=COALESCE(?, dismiss_reason), "
                    "superseded_by=COALESCE(?, superseded_by) WHERE id=?",
                    (r["lifecycle"], now, r.get("reason"), r.get("superseded_by"), r["id"]))
                written += 1

    elif kind == "figure":
        with store.transaction() as conn:
            for r in rows:
                conn.execute(
                    "INSERT INTO figures (key, node_id, path, requirement, caption, "
                    "alt_text, source_url, license, license_url, attribution, status, "
                    "sha256, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "path=COALESCE(excluded.path, figures.path), "
                    "alt_text=COALESCE(excluded.alt_text, figures.alt_text), "
                    "source_url=COALESCE(excluded.source_url, figures.source_url), "
                    "license=COALESCE(excluded.license, figures.license), "
                    "license_url=COALESCE(excluded.license_url, figures.license_url), "
                    "attribution=COALESCE(excluded.attribution, figures.attribution), "
                    "status=excluded.status, sha256=COALESCE(excluded.sha256, figures.sha256), "
                    "updated_at=excluded.updated_at",
                    (r["key"], r.get("node_id"), r.get("path"), r.get("requirement"),
                     r.get("caption"), r.get("alt_text"), r.get("source_url"),
                     r.get("license"), r.get("license_url"), r.get("attribution"),
                     r.get("status", "candidate"), r.get("sha256"), now))
                written += 1

    elif kind == "graph_edge":
        store.add_edges([
            (r["src_type"], r["src_id"], r["rel"], r["dst_type"], r["dst_id"],
             r.get("weight", 1.0), r.get("provenance"))
            for r in rows
        ])
        written = len(rows)

    elif kind == "claim_citation":
        with store.transaction() as conn:
            for r in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO claim_citations (claim_id, bib_key, "
                    "support_status, verified_at, verifier_note) VALUES (?,?,?,?,?)",
                    (r["claim_id"], r["bib_key"], r.get("support_status", "unverified"),
                     now, r.get("note")))
                written += 1

    else:
        raise ToolError(
            f"unknown kb_upsert kind {kind!r}; expected: research_item, evidence, "
            "cluster, lifecycle, figure, graph_edge, claim_citation"
        )

    return {"kind": kind, "written": written}


def _research_row(row: Any) -> dict[str, Any]:
    d = dict(row)
    for field in ("concepts_json", "payload_json"):
        if d.get(field):
            try:
                d[field.replace("_json", "")] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
        d.pop(field, None)
    return d
