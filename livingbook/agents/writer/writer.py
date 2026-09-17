"""Writer Agent.

Runs only after an approved verdict, and only against the files that verdict names.
Produces a patch as an artifact — it does not touch the manuscript until the patch has
cleared verification and QA, so a rejected draft leaves no trace in the book.
"""

from __future__ import annotations

from typing import Any

from ...config import get_config
from ...knowledge.retrieval import BookRetriever
from ...research.models import DraftPatch, ResearchCluster, Verdict, VerdictDecision
from ..base import AgentFailure, BaseAgent


class WriterAgent(BaseAgent[list[DraftPatch]]):
    name = "writer_agent"
    uses_skills = ("book_retrieval", "chapter_editing", "visual_need_detection")

    def degraded_result(self) -> list[DraftPatch]:
        return []

    async def execute(
        self, *, verdict: Verdict, cluster: ResearchCluster, dry_run: bool = True,
        unsupported_claims: list[dict[str, Any]] | None = None,
        technical_findings: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> list[DraftPatch]:
        if not verdict.changes_manuscript:
            raise AgentFailure(
                f"writer invoked for a {verdict.decision.value} verdict, which does "
                "not change the manuscript")
        if not verdict.targets:
            raise AgentFailure("verdict approved but names no target to edit")

        cfg = get_config()
        max_files = int(cfg.get("writer.max_files_per_patch", 3))
        retriever = BookRetriever(self.store)
        editing = self.skill("chapter_editing")

        research_context = self._research_context(cluster, verdict)
        existing_keys = self._bib_keys_in_scope(verdict)

        patches: list[DraftPatch] = []
        for target in verdict.targets[:max_files]:
            if not target.file:
                self.log.warn(f"skipping target with no file: {target.node_id}")
                continue

            surrounding = self._surrounding(retriever, target.node_id)
            instruction = self._instruction(
                verdict, target, cluster,
                unsupported_claims=unsupported_claims or [],
                technical_findings=technical_findings or [])

            patch = await editing(
                self.ctx,
                target_file=f"manuscript/{target.file}",
                node_id=target.node_id,
                instruction=instruction,
                research_context=research_context,
                existing_citations=existing_keys,
                surrounding_context=surrounding,
                decision=verdict.decision.value,
                dry_run=dry_run,
            )
            if patch.is_empty():
                self.log.warn(f"writer produced an empty patch for {target.file}")
                continue

            patches.append(patch)
            self.save("draft_patch", patch,
                      meta={"file": patch.target_file, "node_id": target.node_id,
                            "lines_changed": patch.lines_changed})
            self.log.info(
                f"drafted {patch.lines_changed} lines for {target.file} "
                f"({len(patch.citation_requirements)} citations needed, "
                f"{len(patch.visual_requirements)} figures needed)", status="ok")

        if not patches:
            raise AgentFailure("writer produced no usable patch")
        return patches

    # -- context assembly --------------------------------------------------
    def _research_context(self, cluster: ResearchCluster, verdict: Verdict) -> str:
        """Everything the writer may treat as factual, and nothing else.

        The writer is told elsewhere never to state a fact it cannot cite; this is the
        set of facts it has. Keeping it explicit is what makes that rule checkable.
        """
        lines = [
            f"RESEARCH: {cluster.title}",
            f"MATURITY: {cluster.maturity.value}",
            f"SUMMARY: {cluster.summary}",
            "",
            "EVIDENCE (the ONLY basis for factual statements):",
        ]
        for e in cluster.evidence[:20]:
            provenance = ", ".join(
                f"{p.source}:{p.title[:60]}" for p in e.provenance[:2])
            lines.append(f"  [{e.kind.value}/{e.strength.value}] {e.statement[:400]}")
            if provenance:
                lines.append(f"      source: {provenance}")

        if cluster.claims:
            lines += ["", "KEY CLAIMS:"]
            lines += [f"  - ({c.claim_type.value}) {c.text[:300]}"
                      for c in cluster.claims[:12]]
        if cluster.limitations:
            lines += ["", "LIMITATIONS THE TEXT MUST RESPECT:"]
            lines += [f"  - {x[:250]}" for x in cluster.limitations[:8]]
        if cluster.contradictions:
            lines += ["", "UNRESOLVED DISAGREEMENTS (do not present as settled):"]
            lines += [f"  - {c.statement_a[:140]} VS {c.statement_b[:140]}"
                      for c in cluster.contradictions[:5]]

        lines += ["", "SOURCES:"]
        for p in cluster.provenance[:15]:
            lines.append(f"  - [{p.source}] {p.title[:90]} {p.url}")

        lines += ["", f"VERDICT RATIONALE: {verdict.rationale}"]
        if verdict.claims_needing_evidence:
            lines += ["", "CLAIMS THAT WILL NEED CITATIONS:"]
            lines += [f"  - {c}" for c in verdict.claims_needing_evidence[:10]]
        return "\n".join(lines)

    def _instruction(
        self, verdict: Verdict, target: Any, cluster: ResearchCluster, *,
        unsupported_claims: list[dict[str, Any]] | None = None,
        technical_findings: list[dict[str, Any]] | None = None,
    ) -> str:
        guidance = {
            VerdictDecision.ADD_REFERENCE: (
                "Add a citation to existing content. Do not add new prose beyond a "
                "clause that makes the citation read naturally."),
            VerdictDecision.ADD_FOOTNOTE: (
                "Add a short aside — two or three sentences at most."),
            VerdictDecision.EXTEND_SECTION: (
                "Add one or two paragraphs to the existing section. Do not restructure "
                "it, and do not touch surrounding content."),
            VerdictDecision.ADD_NEW_SECTION: (
                "Add a new subsection with a \\newtag heading and a label in the "
                "existing namespace. It must fit where it is placed."),
            VerdictDecision.REWRITE_SECTION: (
                "Rewrite the identified content because it is now misleading. Preserve "
                "everything still correct; change only what the evidence requires."),
            VerdictDecision.REPLACE_OBSOLETE_CONTENT: (
                "Replace content that is now wrong. State the current position; where "
                "the superseded view is pedagogically useful, keep it as history and "
                "mark it as such."),
        }.get(verdict.decision, "Make the minimal change the verdict describes.")

        # A re-draft has to be told why the last one came back. Without this the Writer
        # receives the identical inputs, writes the identical claims, and the pipeline
        # loops until max_revisions parks it in NEEDS_HUMAN. Measured: one pipeline
        # spent seventeen hours failing to source the same five claims.
        redraft = ""
        if unsupported_claims:
            items = "\n".join(
                f"  - {c.get('claim','')[:200]}\n"
                f"      (rejected: {c.get('why','unsupported')})"
                for c in unsupported_claims[:6])
            redraft += (
                "\n\n=== THIS IS A RE-DRAFT: THE CLAIMS BELOW COULD NOT BE SOURCED ===\n"
                f"{items}\n"
                "The citation search already ran and found nothing supporting these. Do\n"
                "NOT write them again, and do NOT cite a paper that merely repeats the\n"
                "finding — that was already rejected as citation laundering. For each:\n"
                "  - weaken it to what the evidence actually shows, naming the specific\n"
                "    system and setting rather than making a general claim, or\n"
                "  - attribute it in-text as a single reported result, or\n"
                "  - drop it and keep the surrounding prose coherent.\n"
                "A shorter, fully supported passage is the better outcome here.")
        if technical_findings:
            items = "\n".join(
                f"  - {f.get('detail','')[:200]}" for f in technical_findings[:6])
            redraft += (
                "\n\n=== TECHNICAL PROBLEMS IN THE PREVIOUS DRAFT ===\n"
                f"{items}\nFix these specifically; leave what was correct alone.")

        return (
            f"TARGET: {target.section_ref or target.node_id} in {target.file}\n"
            f"CHANGE REQUIRED: {target.change}\n"
            f"ESTIMATED SIZE: about {target.estimated_lines or 20} lines\n\n"
            f"{guidance}\n\n"
            f"CONCEPTS INVOLVED: {', '.join(cluster.concepts[:8])}"
            f"{redraft}"
        )

    def _surrounding(self, retriever: BookRetriever, node_id: str) -> str:
        """The target's own source plus its immediate neighbours.

        The writer needs the neighbours to match voice and to avoid repeating what the
        previous subsection already said — but only the neighbours, not the chapter.
        """
        if not node_id:
            return ""
        ids = [node_id]
        for n in retriever.neighbours_in_reading_order(node_id, window=1):
            ids.append(n.node_id)
        ctx = retriever.context_for(ids, max_tokens=14000, include_parents=True)
        return retriever.render_context(ctx)

    def _bib_keys_in_scope(self, verdict: Verdict) -> list[str]:
        """Citation keys already used nearby, which the writer may reuse freely."""
        keys: list[str] = []
        for target in verdict.targets:
            if not target.node_id:
                continue
            rows = self.store.query(
                "SELECT DISTINCT ce.bib_key FROM cite_edges ce WHERE ce.node_id = ?",
                (target.node_id,))
            keys.extend(r["bib_key"] for r in rows)
        if len(keys) < 25:
            rows = self.store.query(
                "SELECT bib_key FROM cite_edges GROUP BY bib_key "
                "ORDER BY SUM(count) DESC LIMIT 40")
            keys.extend(r["bib_key"] for r in rows)
        return list(dict.fromkeys(keys))[:60]
