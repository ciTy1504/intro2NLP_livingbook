"""Book Verdict Agent — the gate between research and the manuscript.

Two guards run *before* the model is asked anything, and they are deliberately
mechanical:

  * **Evidence floor.** A cluster must clear a minimum of scientific and independent
    evidence before any book change is considered at all.
  * **Maturity gate.** Each decision has a minimum maturity. A speculative finding
    cannot unlock a section rewrite no matter how persuasively it is written up.

Only clusters that pass both are put to the model, and the model's answer is then
re-checked against the same gates — because a confident rationale is exactly what a
plausible-but-premature change looks like.

There is deliberately no numeric threshold that decides a change. `confidence` is
recorded for triage and reporting; the decision rests on the rationale plus the
structural gates.
"""

from __future__ import annotations

import json
from typing import Any

from ...config import get_config
from ...research.models import (
    CitationNeed,
    EditTarget,
    EvidenceKind,
    Maturity,
    ResearchCluster,
    Verdict,
    VerdictDecision,
    VisualRequirement,
)
from ...state.store import utcnow
from ..base import BaseAgent

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": [d.value for d in VerdictDecision]},
        "rationale": {
            "type": "STRING",
            "description": ("Evidence-backed reasoning. Name the specific evidence and "
                            "the specific book content. Not a summary of the research."),
        },
        "scope": {"type": "STRING", "enum": ["minimal", "moderate", "substantial"]},
        "confidence": {"type": "NUMBER"},
        "targets": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "node_id": {"type": "STRING"},
                    "change": {"type": "STRING",
                               "description": "Concretely what to write or alter"},
                    "estimated_lines": {"type": "INTEGER"},
                },
                "required": ["node_id", "change"],
            },
        },
        "affected_content": {"type": "ARRAY", "items": {"type": "STRING"},
                             "description": "node ids whose content this disturbs"},
        "claims_needing_evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
        "citations_required": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                    "preferred_source_type": {"type": "STRING"},
                },
                "required": ["claim"],
            },
        },
        "figures_needed": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "concept": {"type": "STRING"},
                    "purpose": {"type": "STRING"},
                    "expected_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["concept", "purpose"],
            },
        },
        "already_covered": {"type": "BOOLEAN"},
        "risk_if_wrong": {"type": "STRING",
                          "description": "What breaks if this change is premature"},
    },
    "required": ["decision", "rationale", "scope", "confidence"],
}


class BookVerdictAgent(BaseAgent[Verdict]):
    name = "book_verdict_agent"
    uses_skills = ("book_retrieval", "book_placement")

    def degraded_result(self) -> Verdict:
        return Verdict(decision=VerdictDecision.MONITOR,
                       rationale="verdict agent degraded; keeping the cluster under "
                                 "observation rather than acting on it")

    async def execute(self, *, cluster: ResearchCluster, **_: Any) -> Verdict:
        cfg = get_config()

        floor = self._evidence_floor(cluster)
        if not floor["passes"]:
            # Recording this as MONITOR rather than IGNORE is what makes the system
            # track maturity: the same cluster is re-evaluated when new evidence lands.
            verdict = Verdict(
                cluster_id=cluster.id,
                decision=VerdictDecision.MONITOR,
                rationale=(
                    f"Below the evidence floor for a book change. {floor['reason']} "
                    f"Evidence mix: {json.dumps(cluster.evidence_counts())}, "
                    f"{cluster.independent_source_count()} independent source(s). "
                    "Keeping under observation; it will be re-evaluated as evidence "
                    "accumulates."),
                evidence_summary=cluster.evidence_counts(),
                blocked_reason=floor["reason"],
                confidence=0.0,
            )
            await self._persist(cluster, verdict)
            return verdict

        placement = await self.skill("book_placement")(
            self.ctx, title=cluster.title, summary=cluster.summary,
            concepts=cluster.concepts, claims=[c.text for c in cluster.claims])

        allowed = self._allowed_decisions(cluster.maturity)
        data = await self._decide(cluster, placement, allowed, floor)

        verdict = self._build(cluster, placement, data)
        verdict = self._enforce_gates(cluster, verdict, allowed)
        await self._persist(cluster, verdict)

        self.log.info(
            f"verdict on '{cluster.title[:50]}': {verdict.decision.value} "
            f"(maturity={cluster.maturity.value}, scope={verdict.scope})",
            status="ok")
        return verdict

    # -- gates -------------------------------------------------------------
    def _evidence_floor(self, cluster: ResearchCluster) -> dict[str, Any]:
        cfg = get_config()
        need_scientific = int(cfg.get("verdict.min_evidence_for_book_change.scientific", 1))
        need_total = int(cfg.get("verdict.min_evidence_for_book_change.total", 2))

        counts = cluster.evidence_counts()
        scientific = (counts.get(EvidenceKind.SCIENTIFIC.value, 0)
                      + counts.get(EvidenceKind.INDEPENDENT_VERIFICATION.value, 0))
        independent = cluster.independent_source_count()

        if scientific < need_scientific:
            return {"passes": False, "reason": (
                f"Requires at least {need_scientific} scientific or independently "
                f"verified source; found {scientific}.")}
        if independent < need_total:
            return {"passes": False, "reason": (
                f"Requires at least {need_total} independent sources; found "
                f"{independent}.")}
        return {"passes": True, "reason": "", "scientific": scientific,
                "independent": independent}

    def _allowed_decisions(self, maturity: Maturity) -> set[VerdictDecision]:
        """Which decisions this maturity level unlocks."""
        cfg = get_config()
        gates = cfg.get("verdict.maturity_required_for", {}) or {}
        allowed = {VerdictDecision.IGNORE, VerdictDecision.MONITOR}
        for decision, required in gates.items():
            if maturity.value in [str(r) for r in required]:
                try:
                    allowed.add(VerdictDecision(decision))
                except ValueError:
                    continue
        return allowed

    def _enforce_gates(
        self, cluster: ResearchCluster, verdict: Verdict, allowed: set[VerdictDecision],
    ) -> Verdict:
        """Re-apply the gates to the model's answer.

        The prompt states the constraint, but a model asked to pick from a list will
        occasionally pick outside it, and the one case where that happens is the case
        where the research reads as exciting. Downgrading rather than rejecting keeps
        the useful part of the judgement.
        """
        if verdict.decision in allowed:
            return verdict

        fallback = VerdictDecision.MONITOR
        for candidate in (VerdictDecision.EXTEND_SECTION, VerdictDecision.ADD_FOOTNOTE,
                          VerdictDecision.ADD_REFERENCE):
            if candidate in allowed:
                fallback = candidate
                break

        self.log.warn(
            f"verdict {verdict.decision.value} is not permitted at maturity "
            f"{cluster.maturity.value}; downgrading to {fallback.value}")
        verdict.blocked_reason = (
            f"{verdict.decision.value} requires greater maturity than "
            f"{cluster.maturity.value}; downgraded to {fallback.value}")
        verdict.decision = fallback
        if fallback in (VerdictDecision.MONITOR, VerdictDecision.IGNORE):
            verdict.targets = []
            verdict.figures_needed = []
        verdict.scope = "minimal"
        return verdict

    # -- decision ----------------------------------------------------------
    async def _decide(
        self, cluster: ResearchCluster, placement: dict[str, Any],
        allowed: set[VerdictDecision], floor: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = "\n".join(
            f"[{c['node_id']}] {c.get('ref') or c.get('title','')}\n"
            f"   file: {c.get('file')}  summary: {c.get('summary','')[:400]}\n"
            f"   citations: {', '.join(c.get('citations', [])[:8])}"
            for c in (placement.get("candidates") or [])[:10]
        )
        affected = "\n".join(
            f"[{a.get('node_id')}] {a.get('how','')} (severity: {a.get('severity')})"
            for a in (placement.get("affected_sections") or [])[:10]
        )
        evidence_lines = "\n".join(
            f"  [{e.kind.value}/{e.strength.value}] {e.statement[:220]}"
            for e in cluster.evidence[:24]
        )
        contradictions = "\n".join(
            f"  - {c.statement_a[:150]}  VS  {c.statement_b[:150]} ({c.nature})"
            for c in cluster.contradictions[:6]
        )

        prompt = (
            "You decide whether a Vietnamese NLP/LLM textbook should change in response "
            "to new research. Your default answer is NO CHANGE. A textbook that chases "
            "every new result becomes a news feed, and the cost of a premature change is "
            "higher than the cost of a late one.\n\n"
            "=== DECISIONS AVAILABLE TO YOU ===\n"
            f"{', '.join(sorted(d.value for d in allowed))}\n"
            f"(Other decisions are locked because this research is at maturity "
            f"'{cluster.maturity.value}'. Do not choose one that is not listed.)\n\n"
            "  IGNORE                   not relevant, or already adequately covered\n"
            "  MONITOR                  promising but not yet settled enough to teach\n"
            "  ADD_REFERENCE            worth a citation alongside existing content\n"
            "  ADD_FOOTNOTE             worth a short aside\n"
            "  EXTEND_SECTION           add a paragraph or two to an existing section\n"
            "  ADD_NEW_SECTION          genuinely new material the book lacks\n"
            "  REWRITE_SECTION          existing content is now misleading\n"
            "  REPLACE_OBSOLETE_CONTENT existing content is now wrong\n\n"
            "=== YOUR RATIONALE MUST ===\n"
            "  - name the specific evidence, by kind, that justifies the decision\n"
            "  - name the specific book content affected, by node id\n"
            "  - explain why this maturity level warrants this much change\n"
            "  - state what a reader loses if the book stays as it is\n"
            "A rationale that only restates the research is not a rationale.\n\n"
            "=== WEIGH EVIDENCE BY KIND ===\n"
            "scientific and independent_verification carry real weight. adoption and "
            "practical show something works in practice. community and "
            "anecdotal_practical show interest, not truth. Author claims are claims.\n\n"
            f"=== RESEARCH CLUSTER ===\n"
            f"Title: {cluster.title}\n"
            f"Maturity: {cluster.maturity.value}\n"
            f"Summary: {cluster.summary}\n"
            f"Concepts: {', '.join(cluster.concepts)}\n"
            f"Sources: {cluster.independent_source_count()} independent "
            f"({len(cluster.papers)} papers, {len(cluster.implementations)} "
            f"implementations, {len(cluster.benchmarks)} benchmarks, "
            f"{len(cluster.community_signals)} community)\n"
            f"Evidence mix: {json.dumps(cluster.evidence_counts())}\n\n"
            f"EVIDENCE:\n{evidence_lines}\n\n"
            + (f"CONTRADICTIONS:\n{contradictions}\n\n" if contradictions else "")
            + (f"LIMITATIONS:\n  - " + "\n  - ".join(cluster.limitations[:8]) + "\n\n"
               if cluster.limitations else "")
            + "=== WHAT THE BOOK ALREADY SAYS ===\n"
            f"Already covered: {placement.get('already_covered')} "
            f"{placement.get('already_covered_where','')}\n"
            f"Suggested placement: {json.dumps(placement.get('best_location', {}))}\n\n"
            f"CANDIDATE LOCATIONS:\n{candidates}\n\n"
            + (f"CONTENT THIS DISTURBS:\n{affected}\n" if affected else "")
        )

        result = await self.ctx.call(
            "gemini_structured_output", prompt=prompt, schema=VERDICT_SCHEMA,
            temperature=0.15, max_output_tokens=4096)
        return result["data"]

    def _build(
        self, cluster: ResearchCluster, placement: dict[str, Any], data: dict[str, Any],
    ) -> Verdict:
        try:
            decision = VerdictDecision(data.get("decision", "MONITOR"))
        except ValueError:
            decision = VerdictDecision.MONITOR

        targets = []
        for t in (data.get("targets") or [])[:4]:
            node_id = t.get("node_id", "")
            row = self.store.query_one(
                "SELECT file, label, number, kind, title FROM book_nodes WHERE id = ?",
                (node_id,)) if node_id else None
            targets.append(EditTarget(
                node_id=node_id,
                label=row["label"] if row else "",
                file=row["file"] if row else "",
                section_ref=(f"{row['kind']} {row['number']} — {row['title']}"
                             if row else ""),
                change=t.get("change", ""),
                estimated_lines=int(t.get("estimated_lines", 0) or 0),
            ))

        return Verdict(
            cluster_id=cluster.id,
            decision=decision,
            rationale=data.get("rationale", ""),
            scope=data.get("scope", "minimal"),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            targets=targets,
            affected_content=(data.get("affected_content") or [])[:10],
            claims_needing_evidence=(data.get("claims_needing_evidence") or [])[:12],
            citations_required=[
                CitationNeed(claim=c.get("claim", ""), reason=c.get("reason", ""),
                             preferred_source_type=c.get("preferred_source_type", "primary"))
                for c in (data.get("citations_required") or [])[:12]
            ],
            figures_needed=[
                VisualRequirement(
                    concept=f.get("concept", ""), purpose=f.get("purpose", ""),
                    expected_elements=(f.get("expected_elements") or [])[:8])
                for f in (data.get("figures_needed") or [])[:3]
            ],
            evidence_summary=cluster.evidence_counts(),
        )

    async def _persist(self, cluster: ResearchCluster, verdict: Verdict) -> None:
        verdict.id = verdict.id or f"vd_{cluster.id[3:]}"
        self.store.execute(
            "INSERT OR REPLACE INTO verdicts (id, cluster_id, decision, rationale, "
            "scope, confidence, payload_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (verdict.id, cluster.id, verdict.decision.value, verdict.rationale,
             verdict.scope, verdict.confidence,
             json.dumps(verdict.model_dump(mode="json"), ensure_ascii=False), utcnow()),
        )
        self.save("verdict", verdict, meta={"cluster_id": cluster.id,
                                            "decision": verdict.decision.value})

        # Research that led to a change is INTEGRATED; research being watched is
        # MONITOR; research that will never matter is DISMISSED. That lifecycle is
        # what the monitor sweep and the next synthesis pass read.
        lifecycle = ("INTEGRATED" if verdict.changes_manuscript
                     else "MONITOR" if verdict.decision == VerdictDecision.MONITOR
                     else "DISMISSED")
        member_ids = cluster.member_ids()
        if member_ids:
            await self.ctx.call(
                "kb_upsert", kind="lifecycle",
                records=[{"id": mid, "lifecycle": lifecycle,
                          "reason": verdict.blocked_reason or verdict.rationale[:300]}
                         for mid in member_ids])
