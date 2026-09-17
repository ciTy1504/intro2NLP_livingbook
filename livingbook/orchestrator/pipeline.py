"""Pipeline driver: advances one pipeline one state at a time.

Each handler does the work for a single state and returns the next state plus the data
to persist. The transition and the artifact commit together, so a crash resumes at the
last completed state rather than replaying the pipeline — which matters because the
early states cost real money and the late ones touch the repository.

Handlers never call each other. The only way from one state to the next is through
``StateMachine.transition``, which validates the edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..agents import (
    AgentFailure,
    BibtexValidator,
    BookQAAgent,
    BookVerdictAgent,
    CitationAuditor,
    CitationFinder,
    CitationVerifier,
    CrossChapterConsistencyAgent,
    EditorialVerifier,
    EmailAgent,
    EscalateToHuman,
    GitAgent,
    TechnicalVerifier,
    WriterAgent,
)
from ..config import get_config
from ..obs import get_logger, pipeline_scope
from ..research.models import (
    CitationCandidate,
    CitationGap,
    CitationVerification,
    DraftPatch,
    ResearchCluster,
    Verdict,
    VerdictDecision,
)
from ..state.artifacts import get_artifacts
from ..state.machine import Pipeline, State, StateMachine
from ..state.store import get_store
from .visual_flow import run_visual_pipeline


@dataclass
class StepResult:
    next_state: State | None
    note: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None


class PipelineDriver:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.log = get_logger()
        self.store = get_store()
        self.machine = StateMachine(self.store)
        self.artifacts = get_artifacts()

    # -- entry point -------------------------------------------------------
    async def step(self, pipeline: Pipeline) -> Pipeline:
        """Advance one pipeline by exactly one state."""
        handler = self._handlers().get(pipeline.state)
        if handler is None:
            self.log.warn(f"no handler for state {pipeline.state.value}; parking")
            return self.machine.park_for_human(
                pipeline, f"no handler for {pipeline.state.value}")

        with pipeline_scope(pipeline.id, pipeline.state.value):
            self.machine.record_attempt(pipeline)
            try:
                result = await handler(pipeline)
            except EscalateToHuman as exc:
                return self.machine.park_for_human(pipeline, str(exc)[:400])
            except AgentFailure as exc:
                return self.machine.fail(pipeline, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                self.log.error(
                    f"unhandled error in {pipeline.state.value}: "
                    f"{type(exc).__name__}: {exc}")
                return self.machine.fail(pipeline, f"{type(exc).__name__}: {exc}")

        if result.next_state is None:
            return pipeline
        return self.machine.transition(
            pipeline, result.next_state, note=result.note,
            data=result.data, artifact_id=result.artifact_id)

    async def run_to_completion(self, pipeline: Pipeline, *, max_steps: int = 40) -> Pipeline:
        """Drive a pipeline until it terminates, parks, or runs out of steps."""
        for _ in range(max_steps):
            if pipeline.is_terminal or pipeline.is_parked:
                return pipeline
            before = pipeline.state
            pipeline = await self.step(pipeline)
            if pipeline.state == before:
                self.log.warn(f"{pipeline.id} made no progress at {before.value}; stopping")
                return pipeline
        self.log.warn(f"{pipeline.id} hit the step limit at {pipeline.state.value}")
        return pipeline

    def _handlers(self) -> dict[State, Callable[[Pipeline], Awaitable[StepResult]]]:
        return {
            State.SYNTHESIZED: self._verdict_pending,
            State.VERDICT_PENDING: self._verdict,
            State.VERDICT_APPROVED: self._draft,
            State.DRAFTED: self._technical_verify,
            State.TECHNICAL_VERIFY: self._citation_audit,
            State.CITATION_AUDIT: self._citation_find,
            State.CITATION_FIND: self._citation_verify,
            State.CITATION_VERIFY: self._editorial_verify,
            State.EDITORIAL_VERIFY: self._visual,
            State.VISUAL: self._book_qa,
            State.BOOK_QA: self._approve,
            State.APPROVED: self._git_publish,
            State.GIT_PUBLISH: self._email,
            State.EMAIL: self._complete,
        }

    # -- states ------------------------------------------------------------
    async def _verdict_pending(self, pipe: Pipeline) -> StepResult:
        return StepResult(State.VERDICT_PENDING, note="ready for verdict")

    async def _verdict(self, pipe: Pipeline) -> StepResult:
        cluster = self._cluster(pipe)
        if not cluster:
            return StepResult(State.REJECTED, note="cluster missing")

        agent = BookVerdictAgent(pipeline_id=pipe.id)
        verdict: Verdict = await agent.run(cluster=cluster)

        self.store.execute("UPDATE pipelines SET verdict_id = ? WHERE id = ?",
                           (verdict.id, pipe.id))

        if verdict.decision == VerdictDecision.IGNORE:
            return StepResult(State.REJECTED, note=verdict.rationale[:200],
                              data={"verdict": verdict.model_dump(mode="json")})
        if verdict.decision == VerdictDecision.MONITOR:
            return StepResult(State.MONITORING, note=verdict.rationale[:200],
                              data={"verdict": verdict.model_dump(mode="json")})

        # Human approval gate for the decisions that can do the most damage.
        gated = set(self.cfg.get("approval.require_human_approval_for", []) or [])
        if verdict.decision.value in gated:
            self._write_review(pipe, cluster, verdict)
            return StepResult(
                State.NEEDS_HUMAN,
                note=f"{verdict.decision.value} requires human approval",
                data={"verdict": verdict.model_dump(mode="json"),
                      "awaiting_approval": True})

        return StepResult(State.VERDICT_APPROVED, note=verdict.decision.value,
                          data={"verdict": verdict.model_dump(mode="json")})

    async def _draft(self, pipe: Pipeline) -> StepResult:
        cluster = self._cluster(pipe)
        verdict = self._verdict_obj(pipe)
        if not (cluster and verdict):
            return StepResult(State.FAILED, note="missing cluster or verdict")

        # Why the previous draft came back, if it did. Both backward edges into DRAFTED
        # record their reason, and handing it to the Writer is the whole point of
        # sending the work back — without it the re-draft is identical to the draft
        # that was just rejected.
        patches = await self._write_patches(pipe, verdict=verdict, cluster=cluster)

        return StepResult(
            State.DRAFTED,
            note=f"{len(patches)} patch(es), "
                 f"{sum(p.lines_changed for p in patches)} lines",
            data={"patches": [p.model_dump(mode="json") for p in patches],
                  "revision": pipe.data.get("revision", 0)})

    async def _write_patches(
        self, pipe: Pipeline, *, verdict: Any, cluster: Any,
    ) -> list[DraftPatch]:
        """Run the Writer, telling it why the previous draft came back, if it did."""
        agent = WriterAgent(pipeline_id=pipe.id)
        return await agent.run(
            verdict=verdict, cluster=cluster, dry_run=True,
            unsupported_claims=pipe.data.get("unsupported_claims") or [],
            technical_findings=pipe.data.get("technical_findings") or [])

    async def _technical_verify(self, pipe: Pipeline) -> StepResult:
        cluster = self._cluster(pipe)
        patches = self._patches(pipe)
        if not patches:
            return StepResult(State.FAILED, note="no patches to verify")

        # Steps dispatch on the current state, and DRAFTED's step is this one — so
        # _draft, which is the only thing that calls the Writer, runs exactly once,
        # on the way in from VERDICT_APPROVED. A backward edge into DRAFTED therefore
        # re-verified the *same* patches and sent them round again unchanged: five
        # claims, the same five citation gaps, every cycle, until max_revisions parked
        # the pipeline in NEEDS_HUMAN. "Send it back to the Writer" never reached the
        # Writer at all.
        #
        # Pending feedback means the draft in hand is the one that was just rejected.
        # Rewrite it before verifying, and clear the feedback so the next revision
        # does not re-litigate claims already dealt with.
        redraft_data: dict[str, Any] = {}
        pending_claims = pipe.data.get("unsupported_claims") or []
        pending_technical = pipe.data.get("technical_findings") or []
        if pending_claims or pending_technical:
            verdict = self._verdict_obj(pipe)
            if verdict and cluster:
                reason = "citation" if pending_claims else "technical"
                self.log.info(
                    f"re-drafting after {reason} rejection "
                    f"({len(pending_claims)} unsourced claim(s), "
                    f"{len(pending_technical)} technical finding(s))")
                patches = await self._write_patches(
                    pipe, verdict=verdict, cluster=cluster)
                redraft_data = {
                    "patches": [p.model_dump(mode="json") for p in patches],
                    "unsupported_claims": [], "technical_findings": [],
                }
                if not patches:
                    return StepResult(
                        State.NEEDS_HUMAN,
                        note=f"re-draft after {reason} rejection produced no patch",
                        data=redraft_data)

        agent = TechnicalVerifier(pipeline_id=pipe.id)
        reports = []
        for patch in patches:
            reports.append(await agent.run(patch=patch, cluster=cluster))

        blockers = [f for r in reports for f in r.blockers]
        if blockers:
            return StepResult(
                State.DRAFTED,
                note=f"technical blockers: {blockers[0].detail[:150]}",
                data={**redraft_data,
                      "technical_findings": [f.model_dump(mode="json") for f in blockers],
                      "revision": pipe.data.get("revision", 0) + 1,
                      "revision_reason": "technical"})

        return StepResult(
            State.TECHNICAL_VERIFY, note="technical verification passed",
            data={**redraft_data,
                  "technical_reports": [r.model_dump(mode="json") for r in reports]})

    async def _citation_audit(self, pipe: Pipeline) -> StepResult:
        patches = self._patches(pipe)
        verdict = self._verdict_obj(pipe)
        agent = CitationAuditor(pipeline_id=pipe.id)

        gaps: list[CitationGap] = []
        for patch in patches:
            gaps.extend(await agent.run(
                patch=patch,
                affected_node_ids=(verdict.affected_content if verdict else [])[:2]))

        if not gaps:
            # Nothing needs a source, so the finding and verifying states have no work.
            return StepResult(
                State.EDITORIAL_VERIFY, note="no citation gaps",
                data={"citation_gaps": [], "citation_verifications": []})

        return StepResult(
            State.CITATION_AUDIT, note=f"{len(gaps)} citation gap(s)",
            data={"citation_gaps": [g.model_dump(mode="json") for g in gaps]})

    async def _citation_find(self, pipe: Pipeline) -> StepResult:
        gaps = [CitationGap.model_validate(g) for g in pipe.data.get("citation_gaps", [])]
        if not gaps:
            return StepResult(State.CITATION_VERIFY, note="no gaps")

        agent = CitationFinder(pipeline_id=pipe.id)
        candidates = await agent.run(gaps=gaps)
        found = sum(1 for v in candidates.values() if v)

        return StepResult(
            State.CITATION_FIND,
            note=f"candidates for {found}/{len(gaps)} gaps",
            data={"citation_candidates": {
                k: [c.model_dump(mode="json") for c in v] for k, v in candidates.items()}})

    async def _citation_verify(self, pipe: Pipeline) -> StepResult:
        gaps = [CitationGap.model_validate(g) for g in pipe.data.get("citation_gaps", [])]
        raw = pipe.data.get("citation_candidates", {})
        candidates = {
            k: [CitationCandidate.model_validate(c) for c in v] for k, v in raw.items()
        }

        verifier = CitationVerifier(pipeline_id=pipe.id)
        verifications: list[CitationVerification] = await verifier.run(
            gaps=gaps, candidates=candidates)

        validator = BibtexValidator(pipeline_id=pipe.id)
        bib_report = await validator.run(verifications=verifications, write=True)

        accepted = [v for v in verifications if v.verdict == "accept"]
        rejected = [v for v in verifications if v.verdict != "accept"]

        # An unsupported claim is a defect, not an inconvenience: send it back so the
        # Writer softens or removes it rather than shipping it uncited.
        if rejected:
            revision = pipe.data.get("revision", 0) + 1
            unsupported = [
                {"claim": g.claim,
                 "why": next((v.note or v.verdict for v in verifications
                              if v.candidate.title and g.claim), "unsupported")}
                for g in gaps
                if not any(a.candidate.bib_key and _matches(a, g) for a in accepted)
            ]
            if unsupported:
                return StepResult(
                    State.DRAFTED,
                    note=f"{len(unsupported)} claim(s) could not be sourced",
                    data={"unsupported_claims": unsupported,
                          "citation_verifications":
                              [v.model_dump(mode="json") for v in verifications],
                          "bib_report": bib_report,
                          "revision": revision, "revision_reason": "citation"})

        return StepResult(
            State.CITATION_VERIFY,
            note=f"{len(accepted)}/{len(verifications)} citations verified",
            data={"citation_verifications": [v.model_dump(mode="json") for v in verifications],
                  "bib_report": bib_report,
                  "citation_changes": bib_report.get("added", []) +
                                      bib_report.get("reused", [])})

    async def _editorial_verify(self, pipe: Pipeline) -> StepResult:
        patches = self._patches(pipe)
        cluster = self._cluster(pipe)

        editorial = EditorialVerifier(pipeline_id=pipe.id)
        reports = [await editorial.run(patch=p) for p in patches]

        consistency = CrossChapterConsistencyAgent(pipeline_id=pipe.id)
        consistency_report = await consistency.run(
            concepts=(cluster.concepts if cluster else [])[:8],
            focus_node_id=patches[0].node_id if patches else "")

        all_findings = [f for r in reports for f in r.findings] + consistency_report.findings
        blockers = [f for f in all_findings if f.severity == "blocker"]

        if blockers:
            return StepResult(
                State.DRAFTED, note=f"editorial blockers: {blockers[0].detail[:150]}",
                data={"editorial_findings": [f.model_dump(mode="json") for f in blockers],
                      "revision": pipe.data.get("revision", 0) + 1,
                      "revision_reason": "editorial"})

        return StepResult(
            State.EDITORIAL_VERIFY,
            note=f"editorial passed ({len(all_findings)} non-blocking findings)",
            data={"editorial_reports": [r.model_dump(mode="json") for r in reports],
                  "consistency_report": consistency_report.model_dump(mode="json")})

    async def _visual(self, pipe: Pipeline) -> StepResult:
        patches = self._patches(pipe)
        if not self.cfg.get("visual.enabled", True):
            return StepResult(State.VISUAL, note="visual engine disabled",
                              data={"figure_changes": []})

        requirements = [r for p in patches for r in p.visual_requirements]
        if not requirements:
            return StepResult(State.VISUAL, note="no figures required",
                              data={"figure_changes": []})

        placed = await run_visual_pipeline(requirements, pipeline_id=pipe.id)
        return StepResult(
            State.VISUAL, note=f"{len(placed)}/{len(requirements)} figure(s) placed",
            data={"figure_changes": placed})

    async def _book_qa(self, pipe: Pipeline) -> StepResult:
        patches = self._patches(pipe)
        cluster = self._cluster(pipe)

        # QA must run against what would actually ship, so the patches are applied to
        # the working tree first. A failure here reverts them.
        applied = self._apply_patches(patches)
        agent = BookQAAgent(pipeline_id=pipe.id)
        report = await agent.run(
            patch=patches[0] if patches else None,
            concepts=(cluster.concepts if cluster else [])[:8],
            run_build=True)

        if not report.passed:
            self._revert(applied)
            blockers = [f for f in report.semantic + report.global_findings
                        if f.severity == "blocker"]
            det_failures = {
                k: v for k, v in report.deterministic.items()
                if isinstance(v, dict) and not v.get("ok", True)
            }
            return StepResult(
                State.DRAFTED,
                note=f"QA failed: {report.summary[:180]}",
                data={"qa_report": report.model_dump(mode="json"),
                      "qa_failures": {"deterministic": list(det_failures),
                                      "blockers": [b.model_dump(mode="json")
                                                   for b in blockers]},
                      "revision": pipe.data.get("revision", 0) + 1,
                      "revision_reason": "qa"})

        return StepResult(
            State.BOOK_QA, note=report.summary[:200],
            data={"qa_report": report.model_dump(mode="json"),
                  "qa_summary": report.summary,
                  "applied_files": applied})

    async def _approve(self, pipe: Pipeline) -> StepResult:
        return StepResult(State.APPROVED, note="all gates passed")

    async def _git_publish(self, pipe: Pipeline) -> StepResult:
        cluster = self._cluster(pipe)
        patches = self._patches(pipe)
        changed = list(dict.fromkeys(
            [p.target_file for p in patches]
            + [f"manuscript/{f['path']}" for f in pipe.data.get("figure_changes", [])
               if f.get("path")]
            + (["manuscript/references.bib"]
               if pipe.data.get("citation_changes") else [])
        ))

        agent = GitAgent(pipeline_id=pipe.id)
        result = await agent.run(
            pipeline_id=pipe.id, changed_paths=changed,
            topic=(cluster.title if cluster else "update"),
            qa_summary=pipe.data.get("qa_summary", ""))

        if not (result.get("published") or result.get("committed_locally")):
            return StepResult(
                State.NEEDS_HUMAN,
                note=f"publishing failed: {result.get('reason','unknown')}",
                data={"git_result": result})

        next_state = (State.EMAIL if self.cfg.get("email.enabled", True)
                      else State.COMPLETED)
        return StepResult(
            next_state,
            note=f"{result.get('version')} on {result.get('branch')}",
            data={"git_result": result, "version": result.get("version"),
                  "changed_paths": changed})

    async def _email(self, pipe: Pipeline) -> StepResult:
        agent = EmailAgent(pipeline_id=pipe.id)
        result = await agent.run(
            pipeline_id=pipe.id,
            git_result=pipe.data.get("git_result", {}),
            version=pipe.data.get("version", "unversioned"),
            changed_paths=pipe.data.get("changed_paths", []),
            qa_summary=pipe.data.get("qa_summary", ""),
            citation_changes=pipe.data.get("citation_changes", []),
            figure_changes=pipe.data.get("figure_changes", []),
            verification_status=self._verification_status(pipe),
            manuscript_changed=True)
        # A failed notification must not undo a published change.
        return StepResult(State.COMPLETED,
                          note=("emailed" if result.get("sent")
                                else f"email skipped: {result.get('reason','')}"),
                          data={"email_result": result})

    async def _complete(self, pipe: Pipeline) -> StepResult:
        return StepResult(State.COMPLETED, note="done")

    # -- helpers -----------------------------------------------------------
    def _cluster(self, pipe: Pipeline) -> ResearchCluster | None:
        if not pipe.cluster_id:
            return None
        row = self.store.query_one("SELECT payload_json FROM clusters WHERE id = ?",
                                   (pipe.cluster_id,))
        if not row or not row["payload_json"]:
            return None
        try:
            return ResearchCluster.model_validate(json.loads(row["payload_json"]))
        except Exception as exc:
            self.log.warn(f"could not load cluster {pipe.cluster_id}: {exc}")
            return None

    def _verdict_obj(self, pipe: Pipeline) -> Verdict | None:
        blob = pipe.data.get("verdict")
        if blob:
            try:
                return Verdict.model_validate(blob)
            except Exception:
                pass
        if pipe.verdict_id:
            row = self.store.query_one("SELECT payload_json FROM verdicts WHERE id = ?",
                                       (pipe.verdict_id,))
            if row and row["payload_json"]:
                try:
                    return Verdict.model_validate(json.loads(row["payload_json"]))
                except Exception:
                    return None
        return None

    def _patches(self, pipe: Pipeline) -> list[DraftPatch]:
        out = []
        for blob in pipe.data.get("patches", []) or []:
            try:
                out.append(DraftPatch.model_validate(blob))
            except Exception:
                continue
        return out

    def _apply_patches(self, patches: list[DraftPatch]) -> list[dict[str, str]]:
        """Write patches into the working tree, keeping the originals for revert."""
        from ..tools.filesystem import apply_unified_diff

        applied: list[dict[str, str]] = []
        for patch in patches:
            path = self.cfg.root / patch.target_file
            if not path.exists():
                continue
            original = path.read_text(encoding="utf-8", errors="replace")
            try:
                if patch.new_content:
                    updated = patch.new_content
                else:
                    updated = apply_unified_diff(original, patch.unified_diff)
            except Exception as exc:
                self.log.error(f"could not apply patch to {patch.target_file}: {exc}")
                self._revert(applied)
                raise
            path.write_text(updated, encoding="utf-8", newline="\n")
            applied.append({"path": patch.target_file, "original": original})
        return applied

    def _revert(self, applied: list[dict[str, str]]) -> None:
        for entry in applied:
            try:
                (self.cfg.root / entry["path"]).write_text(
                    entry["original"], encoding="utf-8", newline="\n")
            except Exception as exc:
                self.log.error(f"could not revert {entry['path']}: {exc}")

    def _verification_status(self, pipe: Pipeline) -> dict[str, str]:
        data = pipe.data
        qa = data.get("qa_report") or {}
        return {
            "technical": "pass" if data.get("technical_reports") else "n/a",
            "citations": (f"{len([v for v in data.get('citation_verifications', []) if v.get('verdict') == 'accept'])}"
                          f"/{len(data.get('citation_verifications', []))} verified"),
            "editorial": "pass" if data.get("editorial_reports") else "n/a",
            "consistency": "pass" if data.get("consistency_report") else "n/a",
            "build": "pass" if qa.get("build_ok") else "fail",
            "book_qa": "pass" if qa.get("passed") else "fail",
        }

    def _write_review(self, pipe: Pipeline, cluster: ResearchCluster,
                      verdict: Verdict) -> None:
        """Render a human-readable review file for a gated decision."""
        review_dir = self.cfg.path("approval.review_dir", "reviews")
        review_dir.mkdir(parents=True, exist_ok=True)
        path = review_dir / f"{pipe.id}.md"
        lines = [
            f"# Approval required — {verdict.decision.value}",
            "",
            f"**Pipeline**: `{pipe.id}`  ",
            f"**Research**: {cluster.title}  ",
            f"**Maturity**: `{cluster.maturity.value}`  ",
            f"**Scope**: {verdict.scope}",
            "",
            "## Rationale", "", verdict.rationale, "",
            "## Proposed changes", "",
        ]
        for t in verdict.targets:
            lines.append(f"- **{t.section_ref or t.node_id}** (`{t.file}`): {t.change}")
        lines += ["", "## Evidence", ""]
        for e in cluster.evidence[:15]:
            lines.append(f"- `{e.kind.value}`/`{e.strength.value}`: {e.statement[:220]}")
        lines += ["", "## Sources", ""]
        for s in cluster.provenance[:15]:
            lines.append(f"- [{s.source}] [{s.title[:90]}]({s.url})")
        lines += ["", "---", "",
                  "Approve with:", "",
                  f"```\npython -m livingbook.cli approve {pipe.id}\n```", "",
                  "Reject with:", "",
                  f"```\npython -m livingbook.cli reject {pipe.id} --reason \"...\"\n```",
                  ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        self.log.info(f"review written: {path.relative_to(self.cfg.root)}")


def _matches(verification: CitationVerification, gap: CitationGap) -> bool:
    import re
    ta = set(re.findall(r"[a-z0-9]+", gap.claim.lower()))
    tb = set(re.findall(r"[a-z0-9]+", (verification.candidate.title or "").lower()))
    return bool(ta and tb and len(ta & tb) / min(len(ta), len(tb)) > 0.25)
