"""Verification agents: technical, editorial, cross-chapter, full-book QA."""

from __future__ import annotations

from typing import Any

from ...knowledge.retrieval import BookRetriever
from ...research.models import (
    DraftPatch,
    QAReport,
    ResearchCluster,
    VerificationFinding,
    VerificationReport,
)
from ..base import BaseAgent


class TechnicalVerifier(BaseAgent[VerificationReport]):
    name = "technical_verifier"
    uses_skills = ("book_retrieval", "technical_verification")

    def degraded_result(self) -> VerificationReport:
        # A verifier that silently "passes" when it fails is worse than no verifier:
        # it converts an unknown into a false assurance.
        return VerificationReport(
            verifier=self.name, passed=False,
            findings=[VerificationFinding(
                severity="blocker", kind="verifier_unavailable",
                detail="technical verification could not run; the change cannot be "
                       "approved without it")],
            summary="verifier degraded")

    async def execute(
        self, *, patch: DraftPatch, cluster: ResearchCluster | None = None,
        verifications: list[dict[str, Any]] | None = None, **_: Any,
    ) -> VerificationReport:
        text = _added_lines(patch.unified_diff) or patch.new_content
        if not text.strip():
            return VerificationReport(verifier=self.name, passed=True,
                                      summary="no added content to verify")

        research_context = ""
        if cluster:
            research_context = "\n".join(
                f"[{e.kind.value}/{e.strength.value}] {e.statement[:300]}"
                for e in cluster.evidence[:15])

        report = await self.skill("technical_verification")(
            self.ctx, text=text, research_context=research_context,
            location=patch.target_file, verified_citations=verifications or [])

        self.save("verification_report", report,
                  meta={"verifier": self.name, "passed": report.passed,
                        "blockers": len(report.blockers)})
        self.log.info(
            f"technical verification: {'PASS' if report.passed else 'FAIL'} "
            f"({len(report.blockers)} blockers, {len(report.findings)} findings)",
            status="ok" if report.passed else "failed")
        return report


class EditorialVerifier(BaseAgent[VerificationReport]):
    name = "editorial_verifier"
    uses_skills = ("book_retrieval", "editorial_verification")

    def degraded_result(self) -> VerificationReport:
        # Editorial problems are style, not correctness: a degraded run reports a
        # warning rather than blocking a factually sound change.
        return VerificationReport(
            verifier=self.name, passed=True,
            findings=[VerificationFinding(
                severity="note", kind="verifier_unavailable",
                detail="editorial verification did not run")],
            summary="verifier degraded")

    async def execute(self, *, patch: DraftPatch, **_: Any) -> VerificationReport:
        text = _added_lines(patch.unified_diff) or patch.new_content
        if not text.strip():
            return VerificationReport(verifier=self.name, passed=True,
                                      summary="no added content")

        retriever = BookRetriever(self.store)
        summaries, level = "", ""
        if patch.node_id:
            node = retriever._load(patch.node_id)
            if node:
                level = node.summary and "" or ""
                blocks = [f"[{node.node_id}] {node.ref}\n  {node.summary}"]
                for n in retriever.neighbours_in_reading_order(patch.node_id, window=1):
                    blocks.append(f"[{n.node_id}] {n.ref}\n  {n.summary}")
                summaries = "\n".join(blocks)
            row = self.store.query_one(
                "SELECT level FROM node_summaries WHERE node_id = ?", (patch.node_id,))
            level = row["level"] if row else ""

        report = await self.skill("editorial_verification")(
            self.ctx, text=text, location=patch.target_file,
            surrounding_summaries=summaries, level=level or "")

        self.save("verification_report", report,
                  meta={"verifier": self.name, "passed": report.passed})
        self.log.info(
            f"editorial verification: {'PASS' if report.passed else 'FAIL'} "
            f"({len(report.findings)} findings)",
            status="ok" if report.passed else "failed")
        return report


class CrossChapterConsistencyAgent(BaseAgent[VerificationReport]):
    name = "cross_chapter_consistency_agent"
    uses_skills = ("book_retrieval", "cross_chapter_consistency")

    def degraded_result(self) -> VerificationReport:
        return VerificationReport(verifier=self.name, passed=True,
                                  summary="consistency check did not run")

    async def execute(
        self, *, concepts: list[str], focus_node_id: str = "", **_: Any,
    ) -> VerificationReport:
        if not concepts:
            return VerificationReport(verifier=self.name, passed=True,
                                      summary="no concepts to check")

        data = await self.skill("cross_chapter_consistency")(
            self.ctx, concepts=concepts, focus_node_id=focus_node_id)

        findings: list[VerificationFinding] = []
        for c in data.get("contradictions", []):
            findings.append(VerificationFinding(
                severity=c.get("severity", "major"), kind="contradiction",
                detail=(f"{c.get('explanation','')} "
                        f"({c.get('node_a')} vs {c.get('node_b')})"),
                location=f"{c.get('node_a')} / {c.get('node_b')}",
                suggested_fix=""))
        for t in data.get("terminology_drift", []):
            findings.append(VerificationFinding(
                severity="minor", kind="terminology_drift",
                detail=(f"'{t.get('concept')}' appears as "
                        f"{', '.join(t.get('variants', []))}"),
                location=", ".join(t.get("locations", [])[:4]),
                suggested_fix=f"standardise on '{t.get('recommended','')}'"))
        for d in data.get("duplication", []):
            findings.append(VerificationFinding(
                severity="minor", kind="duplication",
                detail=f"{d.get('concept')}: {d.get('explanation','')}",
                location=", ".join(d.get("nodes", [])[:4]),
                suggested_fix=d.get("recommendation", "")))

        passed = not any(f.severity in ("blocker", "major") for f in findings)
        report = VerificationReport(
            verifier=self.name, passed=passed, findings=findings,
            summary=(f"compared {data.get('nodes_compared', 0)} sections; "
                     f"{len(findings)} findings"))
        self.save("verification_report", report,
                  meta={"verifier": self.name, "passed": passed})
        self.log.info(f"cross-chapter consistency: {report.summary}",
                      status="ok" if passed else "failed")
        return report


class BookQAAgent(BaseAgent[QAReport]):
    name = "book_qa_agent"
    uses_skills = ("book_qa", "book_retrieval")

    def degraded_result(self) -> QAReport:
        return QAReport(passed=False, build_ok=False,
                        summary="QA could not run; the change cannot be approved")

    async def execute(
        self, *, patch: DraftPatch | None = None, concepts: list[str] | None = None,
        run_build: bool | None = None, run_global: bool = False,
        check_external_links: bool = False, **_: Any,
    ) -> QAReport:
        patch_text = ""
        if patch:
            patch_text = (f"FILE: {patch.target_file}\n\n"
                          f"{patch.unified_diff[:20000]}")

        report = await self.skill("book_qa")(
            self.ctx, patch_text=patch_text, concepts=concepts or [],
            run_build=run_build, run_global=run_global,
            check_external_links=check_external_links,
            changed_files=[patch.target_file] if patch else None)

        self.save("qa_report", report,
                  meta={"passed": report.passed, "build_ok": report.build_ok,
                        "blockers": report.blocker_count()})
        self.log.info(f"book QA: {'PASS' if report.passed else 'FAIL'} — {report.summary}",
                      status="ok" if report.passed else "failed")
        return report


def _added_lines(diff: str) -> str:
    return "\n".join(
        ln[1:] for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    )
