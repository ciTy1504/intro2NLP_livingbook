"""Citation subsystem agents: auditor, finder, verifier, BibTeX validator.

The chain is built to reject. A claim that cannot be supported by a verified primary
source does not get a weaker citation — it goes back to the Writer to be softened or
cut. That is the only behaviour consistent with a textbook people are meant to trust.
"""

from __future__ import annotations

from typing import Any

from ...config import get_config
from ...research.models import (
    CitationCandidate,
    CitationGap,
    CitationVerification,
    ClaimType,
    DraftPatch,
)
from ..base import BaseAgent


class CitationAuditor(BaseAgent[list[CitationGap]]):
    name = "citation_auditor"
    uses_skills = ("book_retrieval", "citation_audit")

    def degraded_result(self) -> list[CitationGap]:
        return []

    async def execute(
        self, *, patch: DraftPatch | None = None, text: str = "",
        node_id: str = "", location: str = "", include_affected: bool = True,
        affected_node_ids: list[str] | None = None, **_: Any,
    ) -> list[CitationGap]:
        audit = self.skill("citation_audit")
        gaps: list[CitationGap] = []

        if patch:
            added = _added_lines(patch.unified_diff)
            if added.strip():
                existing = self._known_keys(patch.node_id)
                gaps.extend(await audit(
                    self.ctx, text=added, location=patch.target_file,
                    node_id=patch.node_id, existing_citations=existing))

            # A claim the writer already knows it owes evidence for still has to be
            # audited: the writer's self-report is an input, not a substitute.
            for need in patch.citation_requirements:
                if not any(_similar(need.claim, g.claim) for g in gaps):
                    gaps.append(CitationGap(
                        claim=need.claim, node_id=patch.node_id,
                        location=patch.target_file,
                        reason=f"[writer-declared] {need.reason}",
                        preferred_source_type=need.preferred_source_type))

        if text:
            gaps.extend(await audit(
                self.ctx, text=text, location=location, node_id=node_id,
                existing_citations=self._known_keys(node_id)))

        # Existing content the change disturbs is audited too — new material often
        # makes an old unsupported claim newly conspicuous.
        if include_affected and affected_node_ids:
            for nid in affected_node_ids[:3]:
                ctxt = await self.ctx.try_call(
                    "kb_retrieve_context", default=None, node_ids=[nid],
                    max_tokens=6000, include_parents=False)
                if ctxt and ctxt.get("included"):
                    body = ctxt["included"][0].get("content", "")
                    found = await audit(self.ctx, text=body, location=nid, node_id=nid,
                                        existing_citations=self._known_keys(nid))
                    gaps.extend(g for g in found
                                if not any(_similar(g.claim, e.claim) for e in gaps))

        deduped = _dedupe_gaps(gaps)
        if deduped:
            self.save("citation_gap", [g.model_dump(mode="json") for g in deduped],
                      meta={"count": len(deduped)})
        self.log.info(f"citation audit: {len(deduped)} gaps", status="ok")
        return deduped

    def _known_keys(self, node_id: str) -> list[str]:
        if not node_id:
            return []
        rows = self.store.query(
            "SELECT bib_key FROM cite_edges WHERE node_id = ?", (node_id,))
        return [r["bib_key"] for r in rows]


class CitationFinder(BaseAgent[dict[str, list[CitationCandidate]]]):
    name = "citation_finder"
    uses_skills = ("citation_search",)

    def degraded_result(self) -> dict[str, list[CitationCandidate]]:
        return {}

    async def execute(
        self, *, gaps: list[CitationGap], max_per_gap: int | None = None, **_: Any,
    ) -> dict[str, list[CitationCandidate]]:
        cfg = get_config()
        max_per_gap = max_per_gap or int(cfg.get("citation.min_candidates_per_gap", 3))
        search = self.skill("citation_search")

        out: dict[str, list[CitationCandidate]] = {}
        for gap in gaps:
            candidates = await search(self.ctx, gap=gap, max_candidates=max_per_gap + 3)
            out[gap.claim] = candidates
            self.log.info(
                f"found {len(candidates)} candidates for: {gap.claim[:70]}",
                status="ok" if candidates else "empty")

        self.save("citation_candidates",
                  {k: [c.model_dump(mode="json") for c in v] for k, v in out.items()},
                  meta={"gaps": len(gaps),
                        "with_candidates": sum(1 for v in out.values() if v)})
        return out


class CitationVerifier(BaseAgent[list[CitationVerification]]):
    name = "citation_verifier"
    uses_skills = ("citation_verification",)

    def degraded_result(self) -> list[CitationVerification]:
        return []

    async def execute(
        self, *, gaps: list[CitationGap],
        candidates: dict[str, list[CitationCandidate]], max_tries: int = 3, **_: Any,
    ) -> list[CitationVerification]:
        verify = self.skill("citation_verification")
        results: list[CitationVerification] = []

        for gap in gaps:
            pool = candidates.get(gap.claim) or []
            if not pool:
                results.append(_unverifiable(gap, "no candidate sources found"))
                continue

            accepted: CitationVerification | None = None
            attempts: list[CitationVerification] = []
            for candidate in pool[:max_tries]:
                verification = await verify(
                    self.ctx, claim=gap.claim, candidate=candidate,
                    claim_type=gap.claim_type, deep=True)
                attempts.append(verification)
                if verification.verdict == "accept":
                    accepted = verification
                    break
                if verification.laundering_detected:
                    # The source pointed at the real one; that is worth following
                    # rather than discarding.
                    self.log.info(
                        f"citation laundering detected for '{gap.claim[:60]}': "
                        f"{candidate.title[:60]} reports it from "
                        f"{verification.true_primary_source[:80]}")

            chosen = accepted or _best_effort(attempts, gap)
            # The loop knows exactly which gap this answers; recording it here is what
            # lets the pipeline tell a sourced claim from an unsourced one.
            chosen.claim = gap.claim
            results.append(chosen)
            self.log.info(
                f"citation {'ACCEPTED' if chosen.verdict == 'accept' else 'REJECTED'}: "
                f"{gap.claim[:60]} -> {chosen.candidate.title[:50]} ({chosen.verdict})",
                status="ok" if chosen.verdict == "accept" else "rejected")

        accepted_count = sum(1 for r in results if r.verdict == "accept")
        laundered = sum(1 for r in results if r.laundering_detected)
        self.save("citation_verification",
                  [r.model_dump(mode="json") for r in results],
                  meta={"accepted": accepted_count, "total": len(results),
                        "laundering_detected": laundered})
        self.log.info(
            f"citation verification: {accepted_count}/{len(results)} accepted"
            + (f", {laundered} laundering rejections" if laundered else ""),
            status="ok")
        return results


class BibtexValidator(BaseAgent[dict[str, Any]]):
    """The only agent permitted to write references.bib."""

    name = "bibtex_validator"
    uses_skills = ("bibtex_verification",)

    def degraded_result(self) -> dict[str, Any]:
        return {"added": [], "problems": []}

    async def execute(
        self, *, verifications: list[CitationVerification] | None = None,
        validate_existing: bool = False, keys: list[str] | None = None,
        write: bool = True, **_: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"added": [], "reused": [], "refused": [], "problems": []}
        skill = self.skill("bibtex_verification")

        for verification in (verifications or []):
            if verification.verdict != "accept":
                out["refused"].append({
                    "title": verification.candidate.title,
                    "reason": verification.verdict,
                    "note": verification.note,
                })
                continue
            result = await skill(self.ctx, candidate=verification.candidate, write=write)
            if result.get("verdict") == "valid" and result.get("bib_key"):
                verification.candidate.bib_key = result["bib_key"]
                (out["reused"] if result.get("reused") else out["added"]).append({
                    "bib_key": result["bib_key"],
                    "title": verification.candidate.title,
                    "action": "reused" if result.get("reused") else "added",
                })
            else:
                out["refused"].append({
                    "title": verification.candidate.title,
                    "reason": result.get("verdict", "unverifiable"),
                    "note": result.get("note", ""),
                })

        if validate_existing:
            targets = keys or [
                r["bib_key"] for r in self.store.query(
                    "SELECT bib_key FROM bib_entries "
                    "WHERE validation_status = 'unvalidated' LIMIT 40")
            ]
            for key in targets:
                result = await skill(self.ctx, bib_key=key)
                if result.get("verdict") in ("invalid", "suspect"):
                    out["problems"].append(result)
                self.store.execute(
                    "UPDATE bib_entries SET validation_status = ?, validation_note = ?, "
                    "validated_at = datetime('now') WHERE bib_key = ?",
                    (result.get("verdict", "unvalidated"),
                     (result.get("note") or "")[:500], key))

        structural = await self.ctx.try_call("bib_validate", default=None)
        if structural:
            out["structural"] = {
                "ok": structural.get("ok"),
                "orphan_citations": structural.get("orphan_citations", []),
                "errors": [p for p in structural.get("problems", [])
                           if p.get("severity") == "error"],
            }

        self.save("bibtex_report", out,
                  meta={"added": len(out["added"]), "refused": len(out["refused"])})
        self.log.info(
            f"bibtex: {len(out['added'])} added, {len(out['reused'])} reused, "
            f"{len(out['refused'])} refused, {len(out['problems'])} problems",
            status="ok")
        return out


# -- helpers ---------------------------------------------------------------


def _added_lines(diff: str) -> str:
    return "\n".join(
        ln[1:] for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    )


def _similar(a: str, b: str) -> bool:
    import re
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) > 0.7


def _dedupe_gaps(gaps: list[CitationGap]) -> list[CitationGap]:
    out: list[CitationGap] = []
    for g in gaps:
        if not g.claim.strip():
            continue
        if not any(_similar(g.claim, e.claim) for e in out):
            out.append(g)
    return out


def _unverifiable(gap: CitationGap, note: str) -> CitationVerification:
    return CitationVerification(
        candidate=CitationCandidate(title="(none found)"),
        identity_confirmed=False, supports_claim="unverifiable",
        verdict="reject", note=note,
    )


def _best_effort(attempts: list[CitationVerification],
                 gap: CitationGap) -> CitationVerification:
    """Return the most informative rejection, so the Writer knows what to do.

    'needs_primary' with a named true source is far more actionable than a bare
    rejection: the claim may still be supportable, just not by this citation.
    """
    if not attempts:
        return _unverifiable(gap, "no candidates verified")
    for a in attempts:
        if a.verdict == "needs_primary" and a.true_primary_source:
            return a
    for a in attempts:
        if a.supports_claim == "partial":
            return a
    return attempts[0]
