"""MCP server — drive the Living Book from Claude Desktop or Claude Code.

Transport is **stdio**, deliberately. This server needs the local SQLite database, the
local manuscript, the local key pool and the local git repository; a remotely hosted
server could not reach any of them. stdio also means there is nothing to host, nothing
to secure and nothing to keep running — the client spawns it on demand.

What it exposes is a *curated surface*, not a dump of all 62 internal tools. Two
reasons:

  * Most internal tools are pipeline plumbing (``kb_upsert``, ``patch_file``) that only
    make sense inside a state machine that validates their inputs and records their
    provenance. Exposing them would let a chat client corrupt the knowledge base.
  * The interesting operations are questions — "which sections does this research
    affect?", "show me the evidence behind this change" — which map to a handful of
    well-shaped tools rather than to primitives.

Anything that writes runs through the same ``AgentContext`` permission checks as the
autonomous pipeline, so an MCP client cannot do what an agent could not.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import get_config
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.retrieval import BookRetriever
from ..obs import get_logger
from ..state.artifacts import get_artifacts
from ..state.machine import State, StateMachine
from ..state.store import get_store

mcp = MCPServer(
    name="livingbook",
    title="Living Book — intro2NLP",
    version="0.1.0",
    instructions=(
        "Query and operate the Living Book: an autonomous research-to-publication "
        "pipeline around a Vietnamese NLP/LLM textbook.\n\n"
        "Use book_impact to find which parts of the book new research would affect — "
        "that is the query the whole system is built around. Use book_search for "
        "general lookup, book_outline for the whole structure cheaply, and "
        "research_status to see what the pipeline is doing.\n\n"
        "Write operations (approve_change, reject_change) act on a live autonomous "
        "system and are subject to the same permission model as its agents."
    ),
)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# ── reading the book ──────────────────────────────────────────────────────


@mcp.tool(
    title="Find affected book sections",
    description=(
        "Given concepts from a piece of research, return exactly the book sections, "
        "claims and citations that research would affect. This is the core query of "
        "the system: it answers from the knowledge graph without reading the "
        "manuscript, so it is fast and works on a 143k-word book."
    ),
)
async def book_impact(concepts: list[str], limit: int = 10) -> str:
    """Which parts of the book does this research touch?"""
    hits = KnowledgeGraph(get_store()).impact_of_concepts(concepts, limit=limit)
    if not hits:
        return _json({"concepts": concepts, "affected": [],
                      "note": "no section of the book covers these concepts"})
    return _json({
        "concepts": concepts,
        "affected": [
            {
                "section": f"{h.kind} {h.number} — {h.title}",
                "node_id": h.node_id,
                "file": h.file,
                "lines": [h.start_line, h.end_line],
                "score": round(h.score, 3),
                "why": h.reasons,
                "concepts_here": h.concepts[:10],
                "existing_citations": h.citations[:10],
                "claims_at_risk": h.claims[:6],
            }
            for h in hits
        ],
    })


@mcp.tool(
    title="Search the book",
    description="Semantic + concept + lexical search across the manuscript. Returns "
                "ranked sections with summaries, not raw text.",
)
async def book_search(query: str, limit: int = 8) -> str:
    """Find the sections of the book most relevant to a question."""
    hits = await BookRetriever(get_store()).retrieve(query, limit=limit)
    return _json({
        "query": query,
        "hits": [
            {
                "section": h.ref, "node_id": h.node_id, "file": h.file,
                "lines": [h.start_line, h.end_line],
                "summary": h.summary, "concepts": h.concepts[:8],
                "citations": h.citations[:8],
                "score": round(h.score, 3), "matched_by": h.matched_by,
            }
            for h in hits
        ],
    })


@mcp.tool(
    title="Book outline",
    description="The whole table of contents with chapter/section summaries. About "
                "3,300 tokens for the entire book, so it is cheap to include wholesale.",
)
async def book_outline(max_depth: str = "section", with_summaries: bool = True) -> str:
    """The structure of the entire book."""
    return BookRetriever(get_store()).outline(
        max_depth=max_depth, with_summaries=with_summaries)


@mcp.tool(
    title="Read book sections",
    description="Fetch the actual LaTeX source for specific sections, under a token "
                "budget. Pass node_ids from book_impact or book_search.",
)
async def book_read(node_ids: list[str], max_tokens: int = 12000,
                    plain_text: bool = False) -> str:
    """Read the manuscript source for particular sections."""
    retriever = BookRetriever(get_store())
    ctx = retriever.context_for(node_ids, max_tokens=max_tokens, plain_text=plain_text)
    body = retriever.render_context(ctx)
    if ctx.get("dropped"):
        body += f"\n\n[omitted for budget: {', '.join(ctx['dropped'])}]"
    return body


@mcp.tool(
    title="Inspect a citation",
    description="Everywhere a bibliography key is used in the book, what it is claimed "
                "to support, and its verification status.",
)
async def citation_info(bib_key: str) -> str:
    """Where is this source cited, and for what?"""
    return _json(KnowledgeGraph(get_store()).citation_neighbourhood(bib_key))


@mcp.tool(
    title="Concept coverage",
    description="Every section that explains a given concept, in reading order. Useful "
                "for spotting duplicated or scattered treatment.",
)
async def concept_coverage(name: str) -> str:
    """Where does the book explain this concept?"""
    return _json(KnowledgeGraph(get_store()).concept_coverage(name))


# ── the pipeline ──────────────────────────────────────────────────────────


@mcp.tool(
    title="Pipeline status",
    description="What the autonomous system is currently doing: pipelines by state, "
                "research memory by lifecycle, recent verdicts, and anything waiting "
                "for a human decision.",
)
async def research_status() -> str:
    """What is the Living Book doing right now?"""
    store = get_store()
    q = lambda sql, p=(): [dict(r) for r in store.query(sql, p)]
    return _json({
        "pipelines": {r["state"]: r["n"] for r in store.query(
            "SELECT state, COUNT(*) n FROM pipelines GROUP BY state")},
        "research_lifecycle": {r["lifecycle"]: r["n"] for r in store.query(
            "SELECT lifecycle, COUNT(*) n FROM research_items GROUP BY lifecycle")},
        "research_by_source": {r["source"]: r["n"] for r in store.query(
            "SELECT source, COUNT(*) n FROM research_items GROUP BY source ORDER BY n DESC")},
        "clusters_by_maturity": {r["maturity"]: r["n"] for r in store.query(
            "SELECT maturity, COUNT(*) n FROM clusters GROUP BY maturity")},
        "verdicts": {r["decision"]: r["n"] for r in store.query(
            "SELECT decision, COUNT(*) n FROM verdicts GROUP BY decision")},
        "awaiting_human": q(
            "SELECT id, cluster_id, updated_at FROM pipelines "
            "WHERE state = 'NEEDS_HUMAN' ORDER BY updated_at DESC LIMIT 10"),
        "failed": q(
            "SELECT id, last_error, updated_at FROM pipelines "
            "WHERE state = 'FAILED' ORDER BY updated_at DESC LIMIT 5"),
        "scheduler": q("SELECT name, last_run_at, next_run_at, last_status "
                       "FROM scheduled_jobs ORDER BY next_run_at"),
    })


@mcp.tool(
    title="Recent research",
    description="Research the system has discovered, newest first, optionally filtered "
                "by lifecycle (DISCOVERED, MONITOR, INTEGRATED, DISMISSED).",
)
async def research_recent(lifecycle: str = "", limit: int = 20) -> str:
    """What has the system found lately?"""
    store = get_store()
    if lifecycle:
        rows = store.query(
            "SELECT source, title, url, published_at, lifecycle, relevance "
            "FROM research_items WHERE lifecycle = ? "
            "ORDER BY discovered_at DESC LIMIT ?", (lifecycle.upper(), limit))
    else:
        rows = store.query(
            "SELECT source, title, url, published_at, lifecycle, relevance "
            "FROM research_items ORDER BY discovered_at DESC LIMIT ?", (limit,))
    return _json([dict(r) for r in rows])


@mcp.tool(
    title="Explain a decision",
    description="The full evidence chain behind one pipeline: what research drove it, "
                "what evidence supported it, what the verdict decided and why, and "
                "every state it passed through. This is the provenance trail.",
)
async def explain_decision(pipeline_id: str) -> str:
    """Why did the system decide what it decided?"""
    report = get_artifacts().provenance_report(pipeline_id)
    if not report:
        return _json({"error": f"no pipeline {pipeline_id!r}"})
    return _json(report)


@mcp.tool(
    title="Review a pending change",
    description="The proposed change awaiting human approval: the verdict, its "
                "rationale, the evidence, and the actual diff the writer produced.",
)
async def review_pending(pipeline_id: str = "") -> str:
    """Show me what is waiting for my decision."""
    store = get_store()
    machine = StateMachine(store)

    if not pipeline_id:
        pending = machine.in_state(State.NEEDS_HUMAN, limit=10)
        if not pending:
            return _json({"awaiting_human": [], "note": "nothing is waiting for you"})
        pipeline_id = pending[0].id

    pipe = machine.get(pipeline_id)
    data = pipe.data
    verdict = data.get("verdict", {})
    patches = data.get("patches", [])

    return _json({
        "pipeline_id": pipe.id,
        "state": pipe.state.value,
        "revisions": pipe.revisions,
        "verdict": {
            "decision": verdict.get("decision"),
            "scope": verdict.get("scope"),
            "rationale": verdict.get("rationale"),
            "targets": verdict.get("targets"),
        },
        "unsupported_claims": data.get("unsupported_claims", []),
        "patches": [
            {"file": p.get("target_file"), "lines_changed": p.get("lines_changed"),
             "rationale": p.get("rationale"), "diff": p.get("unified_diff", "")[:8000]}
            for p in patches
        ],
        "citation_verifications": [
            {"claim_title": v.get("candidate", {}).get("title"),
             "verdict": v.get("verdict"),
             "laundering": v.get("laundering_detected"),
             "note": v.get("note")}
            for v in data.get("citation_verifications", [])
        ],
        "how_to_act": {
            "approve": f"approve_change('{pipe.id}')",
            "reject": f"reject_change('{pipe.id}', reason='...')",
        },
    })


@mcp.tool(
    title="Approve a pending change",
    description="Release a change that is parked awaiting human approval. It resumes "
                "the pipeline from where it stopped; it does not skip verification.",
    annotations={"destructiveHint": False, "idempotentHint": True},
)
async def approve_change(pipeline_id: str, by: str = "mcp") -> str:
    """Approve a change the system parked for review."""
    from ..orchestrator import Orchestrator

    pipe = await Orchestrator().approve(pipeline_id, by=by)
    get_logger().info(f"{pipeline_id} approved via MCP by {by}")
    return _json({"pipeline_id": pipe.id, "state": pipe.state.value,
                  "note": "the pipeline will continue on the next tick"})


@mcp.tool(
    title="Reject a pending change",
    description="Reject a parked change. The research stays in memory and may be "
                "reconsidered if stronger evidence appears.",
    annotations={"destructiveHint": True},
)
async def reject_change(pipeline_id: str, reason: str = "") -> str:
    """Reject a change the system proposed."""
    from ..orchestrator import Orchestrator

    pipe = await Orchestrator().reject(pipeline_id, reason=reason)
    get_logger().info(f"{pipeline_id} rejected via MCP: {reason[:100]}")
    return _json({"pipeline_id": pipe.id, "state": pipe.state.value, "reason": reason})


@mcp.tool(
    title="Run book QA",
    description="Run quality assurance over the manuscript: deterministic checks "
                "(LaTeX structure, citations, figures) and optionally the full build.",
)
async def run_qa(include_build: bool = False, include_global: bool = False) -> str:
    """Check the book's integrity."""
    from ..agents import BookQAAgent

    report = await BookQAAgent().run(
        run_build=include_build, run_global=include_global)
    return _json({
        "passed": report.passed,
        "summary": report.summary,
        "deterministic": {
            k: {"ok": v.get("ok"), "problems": (v.get("problems") or [])[:10],
                "errors": (v.get("errors") or [])[:10]}
            for k, v in (report.deterministic or {}).items() if isinstance(v, dict)
        },
        "findings": [f.model_dump(mode="json")
                     for f in report.semantic + report.global_findings][:20],
    })


# ── resources ─────────────────────────────────────────────────────────────


@mcp.resource(
    "livingbook://book/outline",
    title="Book outline",
    description="The full table of contents with summaries.",
    mime_type="text/plain",
)
def resource_outline() -> str:
    return BookRetriever(get_store()).outline(max_depth="section")


@mcp.resource(
    "livingbook://architecture",
    title="System architecture",
    description="How the Living Book system is designed and why.",
    mime_type="text/markdown",
)
def resource_architecture() -> str:
    path = get_config().root / "ARCHITECTURE.md"
    return path.read_text(encoding="utf-8") if path.exists() else "(not found)"


@mcp.resource(
    "livingbook://status",
    title="Live system status",
    description="Current pipeline, research and scheduler state.",
    mime_type="application/json",
)
def resource_status() -> str:
    store = get_store()
    return _json({
        "pipelines": {r["state"]: r["n"] for r in store.query(
            "SELECT state, COUNT(*) n FROM pipelines GROUP BY state")},
        "research_items": store.scalar("SELECT COUNT(*) FROM research_items"),
        "clusters": store.scalar("SELECT COUNT(*) FROM clusters"),
        "book_nodes": store.scalar("SELECT COUNT(*) FROM book_nodes"),
    })


# ── prompts ───────────────────────────────────────────────────────────────


@mcp.prompt(
    title="Assess research against the book",
    description="Walk through whether a piece of research should change the textbook.",
)
def assess_research(title: str, summary: str = "") -> str:
    return (
        f"Assess whether this research should change the intro2NLP textbook.\n\n"
        f"Research: {title}\n{summary}\n\n"
        "Steps:\n"
        "1. Call book_impact with the key concepts to see what the book already says.\n"
        "2. Call book_read on the most affected sections to read the actual text.\n"
        "3. Judge whether the book is now wrong, incomplete, or already adequate.\n\n"
        "Remember the system's own standard: a single unreplicated paper is not a "
        "reason to change a textbook. Look for independent verification and real "
        "adoption before recommending a change."
    )


@mcp.prompt(
    title="Review the pending change",
    description="Review what the autonomous system wants to change, and decide.",
)
def review_change() -> str:
    return (
        "Review the change the Living Book is waiting on.\n\n"
        "1. Call review_pending to see the verdict, rationale and diff.\n"
        "2. Call explain_decision on the same pipeline id for the full evidence chain.\n"
        "3. Call book_read on the target section to see the surrounding text.\n\n"
        "Check specifically: does every number in the diff have a verified citation? "
        "Is the Vietnamese consistent with the book's voice? Does it repeat something "
        "the book already says?\n\n"
        "Then recommend approve_change or reject_change, with reasons."
    )


def main() -> None:
    """Entry point. stdio transport: the client spawns this as a subprocess."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
