"""Book retrieval and placement skills.

``BookRetrievalSkill`` is the cheapest way for any agent to find the right part of the
book. ``BookPlacementSkill`` answers the harder question the verdict agent needs: given
new research, *where* does it belong and *what existing content does it disturb* — a
question the knowledge graph can answer structurally before any LLM is involved.
"""

from __future__ import annotations

from typing import Any

from ...knowledge.graph import KnowledgeGraph
from ...knowledge.retrieval import BookRetriever
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, integer, json_schema, number, obj, string

PLACEMENT_SCHEMA = json_schema(
    {
        "best_location": obj({
            "node_id": string("Node id from the candidates, or empty for a new section"),
            "placement": string("", ["extend_existing", "new_subsection", "new_section",
                                     "footnote", "reference_only", "rewrite"]),
            "rationale": string("Why here specifically"),
        }, ["node_id", "placement", "rationale"]),
        "alternatives": array_of(obj({
            "node_id": string(), "why": string(),
        }, ["node_id", "why"])),
        "affected_sections": array_of(obj({
            "node_id": string(),
            "how": string("What in this section is disturbed by the new material"),
            "severity": string("", ["contradicts", "outdates", "extends", "duplicates"]),
        }, ["node_id", "how", "severity"])),
        "prerequisite_gap": string("Concepts the reader would need that the book lacks"),
        "already_covered": boolean("True if the book already says this"),
        "already_covered_where": string(),
    },
    ["best_location", "affected_sections", "already_covered"],
)


class BookRetrievalSkill(Skill):
    """Find the parts of the book relevant to a query, cheapest tier first."""

    name = "book_retrieval"
    required_tools = ("kb_query",)
    optional_tools = ("kb_retrieve_context", "read_file", "gemini_embedding")

    async def run(
        self,
        ctx: AgentContext,
        *,
        query: str = "",
        concepts: list[str] | None = None,
        node_ids: list[str] | None = None,
        limit: int = 8,
        with_source: bool = False,
        max_tokens: int = 20000,
        **_: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if node_ids:
            hits = []
            for nid in node_ids:
                node = await ctx.try_call("kb_query", default=None, kind="node", node_id=nid)
                if node:
                    hits.append(node)
            result["hits"] = hits
        else:
            search = await ctx.call("kb_query", kind="search", query=query,
                                    concepts=concepts, limit=limit)
            result["hits"] = search.get("hits", [])

        if with_source and "kb_retrieve_context" in ctx.allowed_tools:
            ids = node_ids or [h["node_id"] for h in result["hits"][:limit]]
            if ids:
                ctxt = await ctx.call("kb_retrieve_context", node_ids=ids,
                                      max_tokens=max_tokens)
                result["context"] = ctxt
                result["rendered"] = ctxt.get("rendered", "")
        return result

    async def outline(self, ctx: AgentContext, max_depth: str = "section") -> str:
        """The whole book as a few thousand tokens — the global QA tier's input."""
        data = await ctx.call("kb_query", kind="outline", max_depth=max_depth)
        return data.get("outline", "")


class BookPlacementSkill(Skill):
    """Decide where new material belongs and what it disturbs."""

    name = "book_placement"
    required_tools = ("kb_query", "gemini_structured_output")
    optional_tools = ("kb_retrieve_context",)

    async def run(
        self,
        ctx: AgentContext,
        *,
        title: str,
        summary: str,
        concepts: list[str],
        claims: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        # Structural impact first: the graph narrows hundreds of sections to a handful
        # without an LLM call, so the model only ever ranks a shortlist.
        impact = await ctx.call("kb_query", kind="impact", concepts=concepts, limit=10)
        affected = impact.get("affected", [])

        search = await ctx.call(
            "kb_query", kind="search",
            query=f"{title}. {summary}", concepts=concepts, limit=8)
        candidates = {c["node_id"]: c for c in search.get("hits", [])}
        for a in affected:
            candidates.setdefault(a["node_id"], a)

        if not candidates:
            return {
                "best_location": {"node_id": "", "placement": "new_section",
                                  "rationale": "no related content found in the book"},
                "affected_sections": [], "already_covered": False, "candidates": [],
            }

        blocks: list[str] = []
        for c in list(candidates.values())[:12]:
            blocks.append(
                f"[{c['node_id']}] {c.get('ref') or c.get('title', '')}\n"
                f"  file: {c.get('file')} lines {c.get('lines')}\n"
                f"  summary: {c.get('summary', '')[:500]}\n"
                f"  concepts: {', '.join(c.get('concepts', [])[:10])}\n"
                f"  citations: {', '.join(c.get('citations', [])[:8])}\n"
                + (f"  claims: {' | '.join(c.get('claims', [])[:4])}\n"
                   if c.get("claims") else "")
            )

        outline = await ctx.call("kb_query", kind="outline", max_depth="chapter",
                                 with_summaries=False)

        prompt = (
            "You are placing new research inside an existing Vietnamese textbook on "
            "NLP and large language models.\n\n"
            "Decide where it belongs, and — just as important — what existing content "
            "it disturbs. If the book already says this, say so plainly; a textbook "
            "does not improve by repeating itself.\n\n"
            "Prefer the smallest placement that works: extending an existing "
            "subsection beats adding one, which beats adding a section.\n\n"
            f"BOOK OUTLINE:\n{outline.get('outline', '')[:6000]}\n\n"
            f"NEW RESEARCH:\nTitle: {title}\nSummary: {summary}\n"
            f"Concepts: {', '.join(concepts)}\n"
            + (f"Claims:\n- " + "\n- ".join(claims[:8]) if claims else "")
            + f"\n\nCANDIDATE LOCATIONS:\n" + "\n".join(blocks)
        )

        result = await ctx.call(
            "gemini_structured_output", prompt=prompt, schema=PLACEMENT_SCHEMA,
            temperature=0.2)
        data = result["data"]
        data["candidates"] = list(candidates.values())[:12]
        return data
