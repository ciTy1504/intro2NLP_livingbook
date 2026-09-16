"""Chapter editing: turn an approved verdict into a minimal LaTeX patch.

Two properties matter more than prose quality:

  * **Minimality.** The patch must be the smallest coherent change that satisfies the
    verdict. Wholesale rewrites produce diffs nobody can review and destroy the
    provenance link between a research finding and a specific paragraph.
  * **Convention fidelity.** New text must be indistinguishable from the author's.
    The style guide is passed verbatim and the surrounding source is included so the
    model matches the actual file rather than a general idea of LaTeX.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...config import get_config
from ...research.models import Claim, CitationNeed, DraftPatch, VisualRequirement
from ...tools import AgentContext
from ...tools.filesystem import make_unified_diff
from ..base import Skill, array_of, boolean, integer, json_schema, obj, string

EDIT_SCHEMA = json_schema(
    {
        "anchor": string(
            "An EXACT verbatim substring of the existing file, 1-3 lines, that the new "
            "content should be inserted after or that should be replaced. Must appear "
            "exactly once. Leave empty only when appending at the end of the file."),
        "mode": string("", ["insert_after", "replace", "append_end"]),
        "latex": string(
            "The new LaTeX. For insert_after: only the new content. For replace: the "
            "full replacement for the anchor. Vietnamese prose, book conventions."),
        "rationale": string("Why this edit, in English, for the reviewer"),
        "new_claims": array_of(obj({
            "text": string("The claim in English"),
            "claim_type": string("", ["numerical", "benchmark", "historical", "causal",
                                      "sota", "definitional", "architectural",
                                      "attribution"]),
            "needs_citation": boolean(),
        }, ["text", "claim_type", "needs_citation"])),
        "citation_requirements": array_of(obj({
            "claim": string("The exact claim needing a source"),
            "reason": string(),
            "preferred_source_type": string("", ["primary", "original", "official_benchmark",
                                                 "authoritative", "secondary"]),
            "placeholder": string("The \\cite{} placeholder key used in the LaTeX, if any"),
        }, ["claim", "reason"])),
        "visual_requirements": array_of(obj({
            "key": string("snake_case figure key, used as \\bookimage{key}"),
            "concept": string(),
            "purpose": string("What the reader should understand from it"),
            "expected_elements": array_of(string()),
            "relationships": array_of(string()),
            "caption_hint": string("Vietnamese caption text"),
        }, ["key", "concept", "purpose"])),
        "uses_existing_citations": array_of(string(), "bib keys already in the book"),
    },
    ["mode", "latex", "rationale"],
)


class ChapterEditingSkill(Skill):
    name = "chapter_editing"
    required_tools = ("read_file", "gemini_structured_output")
    optional_tools = ("patch_file", "write_file", "kb_retrieve_context", "kb_query",
                      "search_repository")

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()
        guide_path = cfg.root / cfg.get("writer.style_guide", "config/style_guide.md")
        self.style_guide = (
            guide_path.read_text(encoding="utf-8") if guide_path.exists() else ""
        )
        self.max_lines = int(cfg.get("writer.max_patch_lines", 400))

    async def run(
        self,
        ctx: AgentContext,
        *,
        target_file: str,
        node_id: str = "",
        instruction: str,
        research_context: str,
        existing_citations: list[str] | None = None,
        surrounding_context: str = "",
        decision: str = "EXTEND_SECTION",
        dry_run: bool = True,
        **_: Any,
    ) -> DraftPatch:
        file_info = await ctx.call("read_file", path=target_file)
        original = file_info["content"]

        section_source = surrounding_context or original
        if len(section_source) > 30000:
            section_source = section_source[:30000]

        prompt = self._build_prompt(
            target_file=target_file, decision=decision, instruction=instruction,
            research_context=research_context, section_source=section_source,
            existing_citations=existing_citations or [],
        )
        result = await ctx.call(
            "gemini_structured_output", prompt=prompt, schema=EDIT_SCHEMA,
            temperature=0.35, max_output_tokens=8192)
        data = result["data"]

        new_text, applied = self._apply(original, data)
        if not applied:
            raise ValueError(
                f"writer produced an anchor that does not appear exactly once in "
                f"{target_file}; refusing to guess at the location"
            )

        diff = make_unified_diff(original, new_text, target_file)
        lines_changed = sum(
            1 for ln in diff.splitlines()
            if (ln.startswith("+") or ln.startswith("-"))
            and not ln.startswith(("+++", "---"))
        )
        if lines_changed > self.max_lines:
            raise ValueError(
                f"patch is {lines_changed} lines, over the {self.max_lines}-line limit "
                f"for a single verdict. Narrow the verdict or split it."
            )

        if not dry_run and "patch_file" in ctx.allowed_tools:
            await ctx.call("write_file", path=target_file, content=new_text)

        return DraftPatch(
            target_file=target_file,
            node_id=node_id,
            unified_diff=diff,
            new_content=new_text if not dry_run else "",
            rationale=data.get("rationale", ""),
            lines_changed=lines_changed,
            new_claims=[
                Claim(text=c.get("text", ""),
                      claim_type=_claim_type(c.get("claim_type")),
                      needs_citation=bool(c.get("needs_citation", True)))
                for c in (data.get("new_claims") or [])
            ],
            citation_requirements=[
                CitationNeed(claim=c.get("claim", ""), reason=c.get("reason", ""),
                             preferred_source_type=c.get("preferred_source_type", "primary"),
                             suggested_source=c.get("placeholder", ""))
                for c in (data.get("citation_requirements") or [])
            ],
            visual_requirements=[
                VisualRequirement(
                    key=v.get("key", ""), concept=v.get("concept", ""),
                    purpose=v.get("purpose", ""),
                    expected_elements=v.get("expected_elements", []) or [],
                    relationships=v.get("relationships", []) or [],
                    caption_hint=v.get("caption_hint", ""), node_id=node_id)
                for v in (data.get("visual_requirements") or [])
            ],
        )

    # -- prompt ------------------------------------------------------------
    def _build_prompt(
        self, *, target_file: str, decision: str, instruction: str,
        research_context: str, section_source: str, existing_citations: list[str],
    ) -> str:
        return (
            "You are writing for an established Vietnamese textbook on NLP and large "
            "language models. Your text must be indistinguishable from the author's.\n\n"
            "=== STYLE GUIDE (follow exactly) ===\n"
            f"{self.style_guide}\n\n"
            "=== HARD RULES ===\n"
            "1. Write in VIETNAMESE. Only concept names, code and citation keys are English.\n"
            "2. Produce the MINIMAL coherent change. Do not rewrite or reformat anything "
            "the instruction did not ask you to change.\n"
            "3. Never invent a \\cite key. To cite something new, put the claim in "
            "citation_requirements and use \\cite{NEEDS_CITATION} as a placeholder — a "
            "separate pipeline finds and verifies the real source.\n"
            "4. You MAY use these keys, which already exist in the bibliography:\n"
            f"   {', '.join(existing_citations[:60]) if existing_citations else '(none supplied)'}\n"
            "5. Never state a number, benchmark score, date or SOTA claim you cannot "
            "cite. If the research context does not support it, do not write it.\n"
            "6. For figures, emit \\bookimage{key}{detailed Vietnamese description} and "
            "list it in visual_requirements. Never reference an image file directly.\n"
            "7. Mark new headings with \\newtag, per the book's convention.\n"
            "8. The `anchor` must be copied VERBATIM from the source below and must "
            "appear exactly once. This is how the patch is placed — an approximate "
            "anchor will be rejected.\n\n"
            f"=== VERDICT: {decision} ===\n{instruction}\n\n"
            f"=== RESEARCH EVIDENCE (the only basis for factual statements) ===\n"
            f"{research_context[:24000]}\n\n"
            f"=== EXISTING SOURCE: {target_file} ===\n{section_source}\n"
        )

    # -- application -------------------------------------------------------
    def _apply(self, original: str, data: dict[str, Any]) -> tuple[str, bool]:
        mode = data.get("mode", "insert_after")
        latex = (data.get("latex") or "").strip("\n")
        anchor = data.get("anchor") or ""

        if not latex:
            return original, False

        if mode == "append_end" or not anchor:
            return original.rstrip("\n") + "\n\n" + latex + "\n", True

        count = original.count(anchor)
        if count != 1:
            # Retry against a whitespace-normalised view: models reliably reproduce
            # the words of an anchor and unreliably reproduce its exact indentation.
            anchor = self._relocate(original, anchor)
            if anchor is None:
                return original, False
            count = original.count(anchor)
            if count != 1:
                return original, False

        if mode == "replace":
            return original.replace(anchor, latex, 1), True
        return original.replace(anchor, anchor + "\n\n" + latex, 1), True

    @staticmethod
    def _relocate(original: str, anchor: str) -> str | None:
        """Find the real substring corresponding to a whitespace-drifted anchor."""
        target = re.sub(r"\s+", " ", anchor).strip()
        if not target:
            return None
        pattern = re.compile(
            r"\s+".join(re.escape(tok) for tok in target.split(" ") if tok)
        )
        matches = list(pattern.finditer(original))
        if len(matches) != 1:
            return None
        return original[matches[0].start():matches[0].end()]


def _claim_type(value: Any):
    from ...research.models import ClaimType
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.DEFINITIONAL


class CrossChapterConsistencySkill(Skill):
    """Detect contradictions, terminology drift and duplication across chapters."""

    name = "cross_chapter_consistency"
    required_tools = ("kb_query", "gemini_structured_output")
    optional_tools = ("kb_retrieve_context", "gemini_embedding")

    SCHEMA = json_schema(
        {
            "contradictions": array_of(obj({
                "node_a": string(), "node_b": string(),
                "statement_a": string(), "statement_b": string(),
                "explanation": string(),
                "severity": string("", ["blocker", "major", "minor"]),
            }, ["node_a", "node_b", "explanation", "severity"])),
            "terminology_drift": array_of(obj({
                "concept": string(),
                "variants": array_of(string(), "The different terms used"),
                "locations": array_of(string(), "node ids"),
                "recommended": string("Which term the book should standardise on"),
            }, ["concept", "variants", "recommended"])),
            "duplication": array_of(obj({
                "concept": string(),
                "nodes": array_of(string()),
                "explanation": string(),
                "recommendation": string(),
            }, ["concept", "nodes", "explanation"])),
        },
        ["contradictions", "terminology_drift", "duplication"],
    )

    async def run(
        self, ctx: AgentContext, *, concepts: list[str], focus_node_id: str = "",
        max_nodes: int = 10, **_: Any,
    ) -> dict[str, Any]:
        """Compare only sections that share a concept with the change.

        This is what keeps whole-book consistency checking tractable: the graph
        selects the handful of sections that could possibly conflict, and only those
        are read.
        """
        impact = await ctx.call("kb_query", kind="impact", concepts=concepts,
                                limit=max_nodes)
        nodes = impact.get("affected", [])
        if focus_node_id and not any(n["node_id"] == focus_node_id for n in nodes):
            focus = await ctx.try_call("kb_query", default=None, kind="node",
                                       node_id=focus_node_id)
            if focus:
                nodes.insert(0, focus)
        if len(nodes) < 2:
            return {"contradictions": [], "terminology_drift": [], "duplication": [],
                    "nodes_compared": len(nodes)}

        blocks = []
        for n in nodes[:max_nodes]:
            claims = n.get("claims", [])
            blocks.append(
                f"[{n['node_id']}] {n.get('ref') or n.get('title','')}\n"
                f"  summary: {n.get('summary','')[:400]}\n"
                f"  concepts: {', '.join(n.get('concepts', [])[:10])}\n"
                + ("  claims:\n" + "\n".join(f"    - {c}" for c in claims[:8])
                   if claims else "")
            )

        prompt = (
            "Check these sections of one textbook against each other for internal "
            "inconsistency. They were selected because they share concepts.\n\n"
            "Report only genuine problems:\n"
            "  contradictions — two sections assert incompatible things\n"
            "  terminology drift — one concept named differently in different places\n"
            "  duplication — the same explanation given more than once\n\n"
            "Different levels of detail on the same topic are NOT duplication; a "
            "textbook legitimately introduces a concept and later develops it. "
            "Vietnamese and English names for the same concept are NOT drift when the "
            "book's convention is to gloss the English on first use.\n\n"
            f"SECTIONS:\n{chr(10).join(blocks)[:30000]}"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=self.SCHEMA, temperature=0.15)
        data = result["data"]
        data["nodes_compared"] = len(nodes[:max_nodes])
        return data
