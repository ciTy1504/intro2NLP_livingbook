"""Matching an accepted citation back to the claim it was accepted for.

This is the step that decides whether a claim counts as sourced. It compared the
claim text to the paper title by word overlap, which assumed both were English. The
book is Vietnamese. Overlap was 0.00 for every pair, so every accepted citation was
discarded, every claim was reported unsourced no matter how many were accepted, and
the pipeline could not converge at all — it could only exhaust max_revisions and
park in NEEDS_HUMAN.
"""

from livingbook.orchestrator.pipeline import _matches
from livingbook.research.models import (
    CitationCandidate,
    CitationGap,
    CitationVerification,
)


def _verification(claim: str, title: str, verdict: str = "accept"):
    return CitationVerification(
        candidate=CitationCandidate(title=title), claim=claim, verdict=verdict)


def test_a_vietnamese_claim_matches_its_english_source():
    gap = CitationGap(claim="tinh gọn và hợp nhất giao diện công cụ dưới chuẩn MCP",
                      reason="cần nguồn")
    v = _verification(gap.claim,
                      "A Survey on Model Context Protocol: Architecture, Adoption")

    # The two share no words at all — the old comparison scored exactly 0.00.
    assert _matches(v, gap), "an accepted citation must match the claim it answers"


def test_a_verification_does_not_match_a_different_claim():
    gap = CitationGap(claim="tinh gọn và hợp nhất giao diện công cụ dưới chuẩn MCP",
                      reason="cần nguồn")
    other = _verification("Trên T5-XXL (11B), tăng tốc 2--3 lần",
                          "Fast Inference from Transformers via Speculative Decoding")

    assert not _matches(other, gap)


def test_records_without_the_field_still_fall_back():
    """Rows written before the field existed must not all become unsourced."""
    gap = CitationGap(claim="speculative decoding acceptance rate", reason="cần nguồn")
    old = CitationVerification(
        candidate=CitationCandidate(
            title="Speculative decoding acceptance rate in practice"),
        verdict="accept")
    assert old.claim == ""
    assert _matches(old, gap)


def test_the_verifier_records_the_claim_it_verified():
    import inspect

    from livingbook.agents.citation.agents import CitationVerifier

    src = inspect.getsource(CitationVerifier.execute)
    assert "chosen.claim = gap.claim" in src, (
        "the verifier knows which gap it answered and must record it")
