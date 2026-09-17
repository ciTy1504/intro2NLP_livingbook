"""Structured contracts for the research pipeline.

The separation of *signal* from *evidence* is enforced here rather than in prompts.
``EvidenceKind`` is a closed enum, each source agent declares which kinds it may emit
in ``config/agents.yaml``, and ``Evidence.for_source`` refuses anything outside that
set. So a Hacker News thread cannot become scientific evidence by a model deciding it
should be — the type system stops it before the verdict agent ever sees it.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvidenceKind(str, Enum):
    SCIENTIFIC = "scientific"                       # peer-reviewed or preprint result
    PRACTICAL = "practical"                         # works in a real implementation
    ADOPTION = "adoption"                           # the field is actually using it
    COMMUNITY = "community"                         # practitioners are discussing it
    AUTHOR_CLAIM = "author_claim"                   # the authors say so; unverified
    INDEPENDENT_VERIFICATION = "independent_verification"  # someone else reproduced it
    ANECDOTAL_PRACTICAL = "anecdotal_practical"     # one person's report


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ANECDOTAL = "anecdotal"


class Maturity(str, Enum):
    SPECULATIVE = "speculative"
    EMERGING = "emerging"
    CONSOLIDATING = "consolidating"
    MATURE = "mature"
    SUPERSEDED = "superseded"


class Lifecycle(str, Enum):
    DISCOVERED = "DISCOVERED"
    SEEN = "SEEN"
    MONITOR = "MONITOR"
    DISMISSED = "DISMISSED"
    INTEGRATED = "INTEGRATED"
    SUPERSEDED = "SUPERSEDED"


class ClaimType(str, Enum):
    NUMERICAL = "numerical"
    BENCHMARK = "benchmark"
    HISTORICAL = "historical"
    CAUSAL = "causal"
    SOTA = "sota"
    DEFINITIONAL = "definitional"
    ARCHITECTURAL = "architectural"
    ATTRIBUTION = "attribution"


class VerdictDecision(str, Enum):
    IGNORE = "IGNORE"
    MONITOR = "MONITOR"
    ADD_REFERENCE = "ADD_REFERENCE"
    ADD_FOOTNOTE = "ADD_FOOTNOTE"
    EXTEND_SECTION = "EXTEND_SECTION"
    ADD_NEW_SECTION = "ADD_NEW_SECTION"
    REWRITE_SECTION = "REWRITE_SECTION"
    REPLACE_OBSOLETE_CONTENT = "REPLACE_OBSOLETE_CONTENT"


#: Which evidence kinds each source is epistemically capable of producing.
#: A GitHub repository cannot establish a scientific result; a paper can claim one but
#: only independent work verifies it.
SOURCE_EVIDENCE_KINDS: dict[str, set[EvidenceKind]] = {
    "arxiv": {EvidenceKind.SCIENTIFIC, EvidenceKind.AUTHOR_CLAIM},
    "openalex": {EvidenceKind.SCIENTIFIC, EvidenceKind.INDEPENDENT_VERIFICATION},
    "semantic_scholar": {EvidenceKind.SCIENTIFIC, EvidenceKind.INDEPENDENT_VERIFICATION},
    "crossref": {EvidenceKind.SCIENTIFIC},
    "datacite": {EvidenceKind.SCIENTIFIC},
    "openreview": {EvidenceKind.SCIENTIFIC, EvidenceKind.INDEPENDENT_VERIFICATION},
    "acl": {EvidenceKind.SCIENTIFIC, EvidenceKind.INDEPENDENT_VERIFICATION},
    "github": {EvidenceKind.PRACTICAL, EvidenceKind.ADOPTION},
    "huggingface": {EvidenceKind.PRACTICAL, EvidenceKind.ADOPTION},
    "hf_papers": {EvidenceKind.COMMUNITY, EvidenceKind.ADOPTION},
    "blog": {EvidenceKind.PRACTICAL, EvidenceKind.AUTHOR_CLAIM},
    "hackernews": {EvidenceKind.COMMUNITY, EvidenceKind.ANECDOTAL_PRACTICAL},
    "github_discussion": {EvidenceKind.COMMUNITY, EvidenceKind.ANECDOTAL_PRACTICAL},
    "reddit": {EvidenceKind.COMMUNITY, EvidenceKind.ANECDOTAL_PRACTICAL},
    "web": {EvidenceKind.COMMUNITY, EvidenceKind.AUTHOR_CLAIM},
}


class SourceRef(BaseModel):
    """Provenance. Attached to every piece of evidence, never optional."""

    source: str
    source_id: str
    url: str = ""
    title: str = ""
    retrieved_at: str = Field(default_factory=_now)
    published_at: str | None = None
    content_sha256: str | None = None

    def fingerprint(self) -> str:
        return hashlib.sha1(f"{self.source}|{self.source_id}".encode()).hexdigest()[:12]


class BenchmarkResult(BaseModel):
    benchmark: str
    metric: str = ""
    value: str = ""
    model: str = ""
    comparison: str = ""
    is_sota_claim: bool = False


class Claim(BaseModel):
    text: str
    claim_type: ClaimType = ClaimType.DEFINITIONAL
    confidence: float = 0.5
    needs_citation: bool = True
    supporting_sources: list[SourceRef] = Field(default_factory=list)


class Evidence(BaseModel):
    kind: EvidenceKind
    statement: str
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    supports: str | None = None
    provenance: list[SourceRef] = Field(default_factory=list)

    @field_validator("provenance")
    @classmethod
    def _require_provenance(cls, v: list[SourceRef]) -> list[SourceRef]:
        if not v:
            raise ValueError("evidence without provenance is not evidence")
        return v

    @classmethod
    def for_source(
        cls,
        source: str,
        kind: EvidenceKind | str,
        statement: str,
        *,
        strength: EvidenceStrength | str = EvidenceStrength.MODERATE,
        provenance: list[SourceRef],
        supports: str | None = None,
        allowed_kinds: set[EvidenceKind] | None = None,
    ) -> "Evidence":
        """Construct evidence, clamping the kind to what the source can support.

        The clamp rather than a raise is deliberate: a model that labels a forum post
        as ``scientific`` should have its output corrected and recorded, not crash the
        discovery cycle. What must never happen is the mislabel surviving into the
        verdict.
        """
        kind = EvidenceKind(kind)
        permitted = allowed_kinds or SOURCE_EVIDENCE_KINDS.get(
            source, {EvidenceKind.COMMUNITY})
        if kind not in permitted:
            kind = _weakest(permitted)
            strength = EvidenceStrength.WEAK
        return cls(
            kind=kind, statement=statement,
            strength=EvidenceStrength(strength),
            supports=supports, provenance=provenance,
        )


_KIND_RANK = [
    EvidenceKind.ANECDOTAL_PRACTICAL,
    EvidenceKind.COMMUNITY,
    EvidenceKind.AUTHOR_CLAIM,
    EvidenceKind.ADOPTION,
    EvidenceKind.PRACTICAL,
    EvidenceKind.INDEPENDENT_VERIFICATION,
    EvidenceKind.SCIENTIFIC,
]


def _weakest(kinds: set[EvidenceKind]) -> EvidenceKind:
    for k in _KIND_RANK:
        if k in kinds:
            return k
    return EvidenceKind.COMMUNITY


class ResearchItem(BaseModel):
    """One normalised finding from one source."""

    id: str
    source: SourceRef
    title: str
    summary: str = ""
    published_at: str | None = None
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    benchmarks: list[BenchmarkResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    related_work: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    relevance: float = 0.5
    lifecycle: Lifecycle = Lifecycle.DISCOVERED
    raw: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(source: str, source_id: str) -> str:
        digest = hashlib.sha1(f"{source}|{source_id}".encode()).hexdigest()[:12]
        return f"ri_{source}_{digest}"

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.source,
            "source_id": self.source.source_id,
            "url": self.source.url,
            "title": self.title,
            "summary": self.summary,
            "published_at": self.published_at,
            "lifecycle": self.lifecycle.value,
            "relevance": self.relevance,
            "concepts": self.concepts,
            "payload": self.model_dump(mode="json"),
            "content_sha256": self.source.content_sha256,
        }


class Contradiction(BaseModel):
    statement_a: str
    statement_b: str
    source_a: SourceRef | None = None
    source_b: SourceRef | None = None
    nature: str = ""


class ResearchCluster(BaseModel):
    """A coherent topic assembled from many sources. The verdict agent's input."""

    id: str
    title: str
    summary: str = ""
    concepts: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)            # research_item ids
    implementations: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    community_signals: list[str] = Field(default_factory=list)
    blogs: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    maturity: Maturity = Maturity.SPECULATIVE
    provenance: list[SourceRef] = Field(default_factory=list)

    def evidence_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.evidence:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return counts

    def independent_source_count(self) -> int:
        """Distinct sources, so ten arXiv papers do not count as ten channels."""
        return len({p.source for p in self.provenance})

    def member_ids(self) -> list[str]:
        return list(dict.fromkeys(
            self.papers + self.implementations + self.benchmarks
            + self.community_signals + self.blogs
        ))


class EditTarget(BaseModel):
    node_id: str = ""
    label: str = ""
    file: str = ""
    section_ref: str = ""
    change: str = ""
    estimated_lines: int = 0


class CitationNeed(BaseModel):
    claim: str
    reason: str = ""
    preferred_source_type: str = "primary"
    suggested_source: str = ""


class VisualRequirement(BaseModel):
    """What the Writer declares. It never chooses or produces an image itself."""

    key: str = ""
    concept: str
    purpose: str
    expected_elements: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    style: str = "technical diagram, clean, labelled in Vietnamese"
    node_id: str = ""
    caption_hint: str = ""


class Verdict(BaseModel):
    """The gate between research and the manuscript."""

    id: str = ""
    cluster_id: str = ""
    decision: VerdictDecision
    rationale: str
    scope: Literal["minimal", "moderate", "substantial"] = "minimal"
    targets: list[EditTarget] = Field(default_factory=list)
    affected_content: list[str] = Field(default_factory=list)
    claims_needing_evidence: list[str] = Field(default_factory=list)
    citations_required: list[CitationNeed] = Field(default_factory=list)
    figures_needed: list[VisualRequirement] = Field(default_factory=list)
    # Reported for triage and the email report. Never the deciding factor —
    # the gate is the rationale plus the evidence floor in config.
    confidence: float = 0.5
    evidence_summary: dict[str, int] = Field(default_factory=dict)
    blocked_reason: str = ""

    @property
    def changes_manuscript(self) -> bool:
        return self.decision not in (VerdictDecision.IGNORE, VerdictDecision.MONITOR)


class DraftPatch(BaseModel):
    """The Writer's output. A patch, plus what it now owes the rest of the pipeline."""

    target_file: str
    node_id: str = ""
    unified_diff: str = ""
    new_content: str = ""
    rationale: str = ""
    new_claims: list[Claim] = Field(default_factory=list)
    citation_requirements: list[CitationNeed] = Field(default_factory=list)
    visual_requirements: list[VisualRequirement] = Field(default_factory=list)
    lines_changed: int = 0

    def is_empty(self) -> bool:
        return not (self.unified_diff.strip() or self.new_content.strip())


class CitationGap(BaseModel):
    claim: str
    claim_id: str = ""
    location: str = ""
    node_id: str = ""
    reason: str = ""
    claim_type: ClaimType = ClaimType.DEFINITIONAL
    preferred_source_type: str = "primary"


class CitationCandidate(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    abstract: str = ""
    source: SourceRef | None = None
    relevance_rationale: str = ""
    source_tier: Literal["primary", "original", "official_benchmark",
                         "authoritative", "secondary"] = "secondary"
    bib_key: str = ""


class CitationVerification(BaseModel):
    candidate: CitationCandidate
    #: The exact gap.claim this verification answers. Recorded rather than recovered
    #: later by comparing text: the claims are Vietnamese and the titles English, so
    #: word-overlap matching scored 0.00 and silently discarded every accepted
    #: citation in the book.
    claim: str = ""
    identity_confirmed: bool = False
    supports_claim: Literal["supports", "partial", "does_not_support",
                            "laundered", "unverifiable"] = "unverifiable"
    evidence_quote: str = ""
    experimental_setup: str = ""
    reported_numbers: str = ""
    caveats: list[str] = Field(default_factory=list)
    # Citation laundering: the candidate only restates the claim from elsewhere.
    # Rejected even when the claim happens to be true.
    laundering_detected: bool = False
    laundering_note: str = ""
    true_primary_source: str = ""
    verdict: Literal["accept", "reject", "needs_primary"] = "reject"
    note: str = ""


class VerificationFinding(BaseModel):
    severity: Literal["blocker", "major", "minor", "note"] = "minor"
    kind: str = ""
    detail: str
    location: str = ""
    suggested_fix: str = ""


class VerificationReport(BaseModel):
    verifier: str
    passed: bool = False
    findings: list[VerificationFinding] = Field(default_factory=list)
    summary: str = ""

    @property
    def blockers(self) -> list[VerificationFinding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def has_blockers(self) -> bool:
        return bool(self.blockers)


class QAReport(BaseModel):
    deterministic: dict[str, Any] = Field(default_factory=dict)
    semantic: list[VerificationFinding] = Field(default_factory=list)
    global_findings: list[VerificationFinding] = Field(default_factory=list)
    build_ok: bool = False
    passed: bool = False
    summary: str = ""

    def blocker_count(self) -> int:
        return sum(
            1 for f in (self.semantic + self.global_findings) if f.severity == "blocker"
        )
