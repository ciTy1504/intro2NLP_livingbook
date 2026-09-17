"""Technical, editorial and full-book QA skills.

The technical verifier is explicitly instructed to check against retrievable primary
evidence rather than its own recollection — an LLM confidently "remembering" a
benchmark number is the failure mode this whole pipeline exists to prevent.

Book QA is tiered so it never needs the whole manuscript in one context:
deterministic checks run on files, semantic checks run on the patch plus its graph
neighbourhood, and global checks run over summaries and graph projections.
"""

from __future__ import annotations

from typing import Any

from ...config import get_config
from ...research.models import QAReport, VerificationFinding, VerificationReport
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, json_schema, obj, string

FINDINGS_SCHEMA = json_schema(
    {
        "passed": boolean("True only if nothing blocking or major was found"),
        "summary": string("Two sentences for a reviewer"),
        "findings": array_of(obj({
            "severity": string("", ["blocker", "major", "minor", "note"]),
            "kind": string("Short category, e.g. wrong_number, terminology, chronology"),
            "detail": string("What is wrong and why"),
            "location": string("Quote the offending text"),
            "suggested_fix": string(),
        }, ["severity", "kind", "detail"])),
    },
    ["passed", "findings"],
)


class TechnicalVerificationSkill(Skill):
    """Check factual and technical correctness against primary evidence."""

    name = "technical_verification"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("kb_retrieve_context", "kb_query", "read_file", "parse_pdf",
                      "download_pdf", "fetch_url", "fetch_arxiv_paper")

    async def run(
        self, ctx: AgentContext, *, text: str, research_context: str = "",
        location: str = "", verified_citations: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> VerificationReport:
        evidence_block = ""
        if verified_citations:
            lines = []
            for v in verified_citations[:12]:
                lines.append(
                    f"- {v.get('title','')} ({v.get('year','?')}): "
                    f"{v.get('evidence_quote','')[:400]}"
                    + (f"\n  setup: {v.get('experimental_setup','')[:200]}"
                       if v.get("experimental_setup") else "")
                    + (f"\n  numbers: {v.get('reported_numbers','')[:200]}"
                       if v.get("reported_numbers") else "")
                )
            evidence_block = (
                "VERIFIED EVIDENCE (passages actually read from the cited sources — "
                "check the text against THESE, not against your own recollection):\n"
                + "\n".join(lines) + "\n\n"
            )

        prompt = (
            "Verify the technical correctness of this textbook passage. The text is "
            "Vietnamese; answer in English.\n\n"
            "CHECK:\n"
            "  - factual correctness of every technical statement\n"
            "  - algorithm descriptions: are the steps right and in the right order?\n"
            "  - mathematics: are formulas, symbols and dimensions correct?\n"
            "  - model architectures: components, connections, data flow\n"
            "  - benchmark interpretation: does the number mean what the text says?\n"
            "  - chronology: are dates and 'X came before Y' claims right?\n"
            "  - terminology: are standard terms used in their standard sense?\n"
            "  - internal contradictions within the passage\n"
            "  - contradictions with the verified evidence below\n\n"
            "RULES:\n"
            "  - Where verified evidence is supplied, it OUTRANKS your own memory. If "
            "the text disagrees with a quoted passage, the text is wrong.\n"
            "  - Where you are unsure and no evidence is supplied, report a 'note', "
            "not a 'blocker'. Do not invent a correction you cannot support.\n"
            "  - blocker = factually wrong, would mislead a reader.\n"
            "  - major = imprecise or missing a condition that changes the meaning.\n"
            "  - minor = wording.\n\n"
            f"{evidence_block}"
            + (f"RESEARCH CONTEXT:\n{research_context[:12000]}\n\n"
               if research_context else "")
            + f"LOCATION: {location}\n\nTEXT TO VERIFY:\n{text[:28000]}"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=FINDINGS_SCHEMA, temperature=0.05)
        return _to_report("technical_verifier", result["data"])


class EditorialVerificationSkill(Skill):
    """Check pedagogical fit and consistency with the book's voice."""

    name = "editorial_verification"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("kb_retrieve_context", "kb_query", "read_file", "check_language")

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()
        guide = cfg.root / cfg.get("writer.style_guide", "config/style_guide.md")
        self.style_guide = guide.read_text(encoding="utf-8") if guide.exists() else ""

    async def run(
        self, ctx: AgentContext, *, text: str, location: str = "",
        surrounding_summaries: str = "", level: str = "", **_: Any,
    ) -> VerificationReport:
        # Deterministic language check FIRST. A model asked "is this good Vietnamese?"
        # weighs the paragraph as a whole, and a single foreign word inside fluent
        # prose is easy to miss — this verifier passed "cơ chế sinh học của brain
        # humano" on a real patch. A vocabulary check cannot miss it, so it runs
        # regardless of what the model concludes.
        language_findings: list[VerificationFinding] = []
        if "check_language" in ctx.allowed_tools:
            result = await ctx.try_call("check_language", default=None,
                                        text=text, expect="vi")
            if result:
                for f in result.get("findings", []):
                    language_findings.append(VerificationFinding(
                        severity=f["severity"],
                        kind=f["kind"],
                        detail=f["detail"],
                        location=f.get("context", "")[:300],
                        suggested_fix=(
                            f"remove or translate {f['word']!r}"
                            if f["kind"] == "foreign_word"
                            else f"confirm {f['word']!r} is intended; gloss it on first use"
                        ),
                    ))

        prompt = (
            "Review this new textbook passage for editorial and pedagogical fit. "
            "The book is a Vietnamese graduate-level text on NLP and LLMs. Answer in "
            "English; quote Vietnamese text when pointing at a problem.\n\n"
            "CHECK:\n"
            "  - LANGUAGE PURITY: every word must be Vietnamese, an established English "
            "technical term, or LaTeX. Flag as a BLOCKER any word from another language "
            "(Spanish, Portuguese, French, Italian) that has drifted into the prose — "
            "this has happened before and is easy to read past.\n"
            "  - level: does it match the surrounding material, or assume too much/little?\n"
            "  - prerequisites: does it use concepts the book has not introduced?\n"
            "  - clarity: would a reader at this point actually follow it?\n"
            "  - terminology: consistent with the book's existing Vietnamese terms, with "
            "the English glossed on first use?\n"
            "  - transitions: does it join the surrounding text, or read as pasted in?\n"
            "  - duplication: does it repeat something the neighbouring sections say?\n"
            "  - unnecessary complexity: is anything more complicated than it needs to be?\n"
            "  - style-guide compliance (below)\n\n"
            "Do NOT flag: correct technical content you find difficult, use of English "
            "technical terms the guide permits, or length alone.\n\n"
            f"=== STYLE GUIDE ===\n{self.style_guide[:8000]}\n\n"
            + (f"=== SURROUNDING SECTIONS ===\n{surrounding_summaries[:8000]}\n\n"
               if surrounding_summaries else "")
            + (f"Expected level: {level}\n" if level else "")
            + f"LOCATION: {location}\n\n=== NEW TEXT ===\n{text[:24000]}"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=FINDINGS_SCHEMA, temperature=0.15)
        report = _to_report("editorial_verifier", result["data"])

        # The deterministic findings are authoritative: they are merged in and can
        # fail the report even when the model was satisfied.
        if language_findings:
            report.findings.extend(language_findings)
            if any(f.severity in ("blocker", "major") for f in language_findings):
                report.passed = False
                blockers = [f.detail for f in language_findings
                            if f.severity == "blocker"]
                report.summary = (
                    f"language check failed: {'; '.join(blockers[:2])}. "
                    + report.summary)
        return report


class BookQASkill(Skill):
    """Three-tier QA: deterministic, semantic, global."""

    name = "book_qa"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("build_book", "run_tests", "latex_lint", "bib_validate",
                      "check_links", "check_figures", "check_language",
                      "kb_query", "kb_retrieve_context")

    async def run(
        self, ctx: AgentContext, *, changed_files: list[str] | None = None,
        patch_text: str = "", concepts: list[str] | None = None,
        run_build: bool | None = None, run_global: bool = False,
        check_external_links: bool = False, **_: Any,
    ) -> QAReport:
        cfg = get_config()
        report = QAReport()

        report.deterministic = await self._deterministic(
            ctx, run_build=cfg.get("qa.run_build", True) if run_build is None else run_build,
            check_links=check_external_links,
            patch_text=patch_text,
        )
        report.build_ok = bool(report.deterministic.get("build", {}).get("ok", True))

        if patch_text:
            report.semantic = await self._semantic(ctx, patch_text, concepts or [])
        if run_global:
            report.global_findings = await self._global(ctx)

        det_ok = all(
            v.get("ok", True) for k, v in report.deterministic.items()
            if isinstance(v, dict) and k != "links"
        )
        report.passed = det_ok and report.blocker_count() == 0
        report.summary = self._summarise(report)
        return report

    # -- tier 1: deterministic --------------------------------------------
    async def _deterministic(
        self, ctx: AgentContext, *, run_build: bool, check_links: bool,
        patch_text: str = "",
    ) -> dict[str, Any]:
        """No LLM. Fast, objective, and able to fail a pipeline on facts."""
        out: dict[str, Any] = {}
        for tool_name, key in (("latex_lint", "latex"), ("bib_validate", "bibliography"),
                               ("check_figures", "figures")):
            if tool_name in ctx.allowed_tools:
                out[key] = await ctx.try_call(
                    tool_name, default={"ok": True, "skipped": True})

        # Language purity, on the added lines only. Checking the whole manuscript
        # would flag the author's own established vocabulary, which is the baseline
        # this check is measured against.
        if patch_text and "check_language" in ctx.allowed_tools:
            added = "\n".join(
                ln[1:] for ln in patch_text.splitlines()
                if ln.startswith("+") and not ln.startswith("+++"))
            if added.strip():
                out["language"] = await ctx.try_call(
                    "check_language", default={"ok": True, "skipped": True},
                    text=added, expect="vi")
        if check_links and "check_links" in ctx.allowed_tools:
            # Informational: a dead external link is a real defect but not one that
            # should block a manuscript change, since the web breaks on its own.
            out["links"] = await ctx.try_call(
                "check_links", default={"ok": True, "skipped": True}, limit=40)
        if run_build and "build_book" in ctx.allowed_tools:
            out["build"] = await ctx.try_call(
                "build_book", default={"ok": False, "error": "build unavailable"})
        return out

    # -- tier 2: semantic --------------------------------------------------
    async def _semantic(
        self, ctx: AgentContext, patch_text: str, concepts: list[str],
    ) -> list[VerificationFinding]:
        """Scoped to the patch plus the sections that share its concepts."""
        neighbourhood = ""
        if concepts and "kb_query" in ctx.allowed_tools:
            impact = await ctx.try_call("kb_query", default={}, kind="impact",
                                        concepts=concepts, limit=8)
            blocks = []
            for n in impact.get("affected", [])[:8]:
                blocks.append(
                    f"[{n['node_id']}] {n.get('ref') or n.get('title','')}\n"
                    f"  {n.get('summary','')[:400]}\n"
                    f"  concepts: {', '.join(n.get('concepts', [])[:8])}")
            neighbourhood = "\n".join(blocks)

        prompt = (
            "Quality-check a change to a Vietnamese NLP/LLM textbook, against the "
            "sections it is most likely to interact with.\n\n"
            "CHECK: terminology consistency, whether claims are supported by their "
            "citations, whether cross-references point somewhere sensible, "
            "contradictions with neighbouring sections, duplicated explanation, and "
            "whether the chapter still flows across the seam.\n\n"
            "Answer in English. Report only real problems.\n\n"
            + (f"=== NEIGHBOURING SECTIONS ===\n{neighbourhood[:14000]}\n\n"
               if neighbourhood else "")
            + f"=== THE CHANGE ===\n{patch_text[:24000]}"
        )
        result = await ctx.try_call("gemini_structured_output", default=None,
                                    prompt=prompt, schema=FINDINGS_SCHEMA, temperature=0.1)
        if not result:
            return []
        return _to_report("book_qa_semantic", result["data"]).findings

    # -- tier 3: global ----------------------------------------------------
    async def _global(self, ctx: AgentContext) -> list[VerificationFinding]:
        """Whole-book coherence from summaries and graph projections only.

        The outline plus concept and citation statistics is a few hundred KB whatever
        the manuscript's size, which is what makes a whole-book check affordable at all.
        """
        outline = await ctx.try_call("kb_query", default={}, kind="outline",
                                     max_depth="section")
        concepts = await ctx.try_call("kb_query", default={}, kind="concepts", limit=120)
        stats = await ctx.try_call("kb_query", default={}, kind="stats")

        concept_lines = "\n".join(
            f"  {c['name']} — explained in {c['sections']} sections"
            for c in (concepts.get("concepts") or [])[:120]
        )
        prompt = (
            "Assess the global coherence of a Vietnamese NLP/LLM textbook using its "
            "structure and concept index. You are NOT reading the full text; judge only "
            "what these projections can support.\n\n"
            "LOOK FOR:\n"
            "  - a concept explained in many separate places (possible duplication)\n"
            "  - near-duplicate concept names that are probably one concept under two "
            "terms (terminology drift)\n"
            "  - a chapter whose sections do not match its title\n"
            "  - obvious gaps: a concept relied on everywhere but introduced nowhere\n"
            "  - ordering problems: a concept used well before it is introduced\n\n"
            "Be conservative — do not report a problem the outline cannot actually "
            "demonstrate. Answer in English.\n\n"
            f"=== STATS ===\n{stats}\n\n"
            f"=== CONCEPT INDEX ===\n{concept_lines[:12000]}\n\n"
            f"=== OUTLINE ===\n{outline.get('outline','')[:20000]}"
        )
        result = await ctx.try_call("gemini_structured_output", default=None,
                                    prompt=prompt, schema=FINDINGS_SCHEMA, temperature=0.1)
        if not result:
            return []
        return _to_report("book_qa_global", result["data"]).findings

    @staticmethod
    def _summarise(report: QAReport) -> str:
        parts = []
        for key, value in report.deterministic.items():
            if isinstance(value, dict):
                mark = "ok" if value.get("ok", True) else "FAIL"
                parts.append(f"{key}={mark}")
        blockers = report.blocker_count()
        parts.append(f"semantic_findings={len(report.semantic)}")
        parts.append(f"global_findings={len(report.global_findings)}")
        parts.append(f"blockers={blockers}")
        return ", ".join(parts)


def _to_report(name: str, data: dict[str, Any]) -> VerificationReport:
    findings = [
        VerificationFinding(
            severity=f.get("severity", "minor"),
            kind=f.get("kind", ""),
            detail=f.get("detail", ""),
            location=(f.get("location") or "")[:500],
            suggested_fix=(f.get("suggested_fix") or "")[:800],
        )
        for f in (data.get("findings") or [])
    ]
    passed = bool(data.get("passed")) and not any(
        f.severity in ("blocker", "major") for f in findings)
    return VerificationReport(
        verifier=name, passed=passed, findings=findings,
        summary=data.get("summary", ""),
    )
