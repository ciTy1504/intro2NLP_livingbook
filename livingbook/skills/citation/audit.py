"""Citation auditing, search and verification.

The citation subsystem is where most of this system's credibility lives, so it is
built to reject rather than to accept. Three properties:

  * The **finder** searches for evidence for the *claim*, not for text resembling the
    claim. A keyword match on a title is not evidence.
  * The **verifier** reads the source and checks it actually supports the claim, and
    explicitly hunts for citation laundering — where paper A merely reports that paper
    B claims X, and the book then cites A for X.
  * Anything unverifiable is rejected, and the claim it was meant to support goes
    back to the Writer to be softened or removed. A textbook is better with one fewer
    sentence than with one unsupported number.
"""

from __future__ import annotations

import re
from typing import Any

from ...config import get_config
from ...research.models import (
    CitationCandidate,
    CitationGap,
    CitationVerification,
    ClaimType,
    SourceRef,
)
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, integer, json_schema, number, obj, string

AUDIT_SCHEMA = json_schema(
    {
        "gaps": array_of(obj({
            "claim": string("The exact claim from the text, quoted or closely paraphrased"),
            "reason": string("Why this needs a source it does not have"),
            "claim_type": string("", ["numerical", "benchmark", "historical", "causal",
                                      "sota", "definitional", "architectural",
                                      "attribution"]),
            "preferred_source_type": string("", ["primary", "original",
                                                 "official_benchmark", "authoritative",
                                                 "secondary"]),
            "severity": string("", ["blocker", "major", "minor"]),
            "existing_citation": string("bib key currently attached, if any"),
            "problem": string("", ["missing", "weak", "outdated", "wrong_source",
                                   "unsupported"]),
        }, ["claim", "reason", "claim_type", "preferred_source_type", "severity",
            "problem"])),
        "well_supported": array_of(string(), "Claims that already have adequate sources"),
    },
    ["gaps"],
)

RANK_SCHEMA = json_schema(
    {
        "ranked": array_of(obj({
            "index": integer("Index into the candidate list"),
            "source_tier": string("", ["primary", "original", "official_benchmark",
                                       "authoritative", "secondary"]),
            "supports_claim_likelihood": number("0..1"),
            "rationale": string("Why this source would or would not settle the claim"),
        }, ["index", "source_tier", "supports_claim_likelihood", "rationale"])),
        "best_index": integer("Index of the best candidate, or -1 if none is adequate"),
        "note": string(),
    },
    ["ranked", "best_index"],
)

VERIFY_SCHEMA = json_schema(
    {
        "identity_confirmed": boolean("Do title/authors/year/venue match the candidate?"),
        "supports_claim": string("", ["supports", "partial", "does_not_support",
                                      "laundered", "unverifiable"]),
        "evidence_quote": string("The passage that supports the claim, quoted verbatim"),
        "experimental_setup": string("Conditions under which the result holds"),
        "reported_numbers": string("The exact numbers reported, if the claim is numeric"),
        "caveats": array_of(string(), "Conditions the book must state alongside the claim"),
        "laundering_detected": boolean(
            "True if this source only REPORTS the claim from another work rather than "
            "establishing it"),
        "laundering_note": string(),
        "true_primary_source": string("The work that actually establishes the claim"),
        "verdict": string("", ["accept", "reject", "needs_primary"]),
        "note": string("One sentence for the reviewer"),
    },
    ["identity_confirmed", "supports_claim", "laundering_detected", "verdict"],
)


class CitationAuditSkill(Skill):
    """Find claims that need evidence they do not have."""

    name = "citation_audit"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("kb_retrieve_context", "kb_query", "read_file")

    async def run(
        self, ctx: AgentContext, *, text: str, location: str = "", node_id: str = "",
        existing_citations: list[str] | None = None, **_: Any,
    ) -> list[CitationGap]:
        cfg = get_config()
        strict_types = cfg.get("citation.require_primary_source_for",
                               ["numerical", "benchmark", "sota", "causal"])

        bib_context = ""
        if existing_citations and "kb_query" in ctx.allowed_tools:
            lines = []
            for key in existing_citations[:30]:
                entry = await ctx.try_call("kb_query", default=None, kind="bib", bib_key=key)
                if entry:
                    lines.append(f"  {key}: {entry.get('title','')} "
                                 f"({entry.get('year','?')}, {entry.get('venue','')})")
            bib_context = "\n".join(lines)

        prompt = (
            "Audit this textbook passage for claims that need evidence.\n\n"
            "The text is Vietnamese; answer in English. Citations appear as "
            "\\cite{key} or [key].\n\n"
            "FLAG a claim when it is:\n"
            "  - a specific number, measurement or benchmark score\n"
            "  - a state-of-the-art or 'best/fastest/largest' assertion\n"
            "  - a historical or chronological statement (who did what, when)\n"
            "  - a causal claim ('X improves Y because Z')\n"
            "  - an attribution of a method to particular authors\n"
            "  - a claim whose attached citation looks wrong, weak or outdated\n\n"
            "DO NOT flag: definitions of standard terms, pedagogical framing, worked "
            "examples with invented numbers, statements about the book's own structure, "
            "or widely known facts a textbook may assert without a source "
            "(e.g. 'the Transformer uses self-attention').\n\n"
            f"These claim types REQUIRE a primary source: {', '.join(strict_types)}.\n\n"
            + (f"CITATIONS CURRENTLY AVAILABLE:\n{bib_context}\n\n" if bib_context else "")
            + f"LOCATION: {location}\n\nTEXT:\n{text[:28000]}"
        )

        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=AUDIT_SCHEMA, temperature=0.1)
        gaps = []
        for g in result["data"].get("gaps", []) or []:
            gaps.append(CitationGap(
                claim=g.get("claim", ""),
                location=location,
                node_id=node_id,
                reason=f"[{g.get('problem', 'missing')}/{g.get('severity','major')}] "
                       f"{g.get('reason', '')}",
                claim_type=_claim_type(g.get("claim_type")),
                preferred_source_type=g.get("preferred_source_type", "primary"),
            ))
        return gaps


class CitationSearchSkill(Skill):
    """Find sources that support the exact claim, preferring primary ones."""

    name = "citation_search"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("search_openalex", "search_arxiv", "search_crossref",
                      "search_semantic_scholar", "search_acl_anthology", "web_search")

    async def run(
        self, ctx: AgentContext, *, gap: CitationGap, max_candidates: int = 8, **_: Any,
    ) -> list[CitationCandidate]:
        queries = await self._queries_for(ctx, gap)
        raw: list[dict[str, Any]] = []

        cfg = get_config()
        # Skip sources the configuration has switched off rather than calling them and
        # logging a failure per query — an unattended run should not fill its log with
        # the same "this is disabled" notice dozens of times a cycle.
        disabled = {
            "search_semantic_scholar": not cfg.get(
                "research.sources.semantic_scholar.enabled", False),
        }

        for query in queries[:3]:
            for tool_name, kwargs in (
                ("search_openalex", {"query": query, "per_page": 6}),
                ("search_arxiv", {"query": query, "max_results": 6}),
                ("search_crossref", {"query": query, "rows": 4}),
                ("search_acl_anthology", {"query": query, "limit": 4}),
                ("search_semantic_scholar", {"query": query, "limit": 4}),
            ):
                if tool_name not in ctx.allowed_tools or disabled.get(tool_name):
                    continue
                found = await ctx.try_call(tool_name, default=[], **kwargs)
                raw.extend(found or [])

        deduped = _dedupe_candidates(raw)
        if not deduped:
            return []

        ranked = await self._rank(ctx, gap, deduped[:16])
        return ranked[:max_candidates]

    async def _queries_for(self, ctx: AgentContext, gap: CitationGap) -> list[str]:
        """Turn a claim into search queries aimed at the evidence, not the wording.

        Searching the claim verbatim finds papers that *phrase things similarly*.
        Searching for the study that would establish it finds the source that settles
        it, which is the entire point of the distinction in the brief.
        """
        schema = json_schema(
            {"queries": array_of(string(), "3 search queries, most specific first"),
             "what_would_prove_it": string("What kind of study or document settles this")},
            ["queries"])
        prompt = (
            "A textbook makes this claim and needs a source that ESTABLISHES it.\n\n"
            f"CLAIM: {gap.claim}\n"
            f"TYPE: {gap.claim_type.value}\n"
            f"WHY IT NEEDS A SOURCE: {gap.reason}\n"
            f"PREFERRED SOURCE: {gap.preferred_source_type}\n\n"
            "Write search queries that would find the work that actually establishes "
            "this — the original paper, the benchmark's own publication, the study that "
            "measured it. Do not simply restate the claim as a query; that finds papers "
            "with similar wording rather than the source of the fact.\n"
            "Use the standard English technical terms researchers would use."
        )
        result = await ctx.try_call("gemini_structured_output", default=None,
                                    prompt=prompt, schema=schema, role="fast",
                                    temperature=0.2)
        if result and result["data"].get("queries"):
            return [q for q in result["data"]["queries"] if q]
        return [gap.claim[:200]]

    async def _rank(
        self, ctx: AgentContext, gap: CitationGap, candidates: list[dict[str, Any]],
    ) -> list[CitationCandidate]:
        listing = "\n\n".join(
            f"[{i}] {c.get('title','')}\n"
            f"    authors: {', '.join(str(a) for a in (c.get('authors') or [])[:6])}\n"
            f"    year: {c.get('year') or c.get('published','')}  "
            f"venue: {c.get('venue','')}  citations: {c.get('citation_count','?')}\n"
            f"    doi: {c.get('doi','')}  arxiv: {c.get('arxiv_id','')}\n"
            f"    abstract: {(c.get('abstract') or '')[:700]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Rank these candidate sources by how well each would SUPPORT the specific "
            "claim below — not by how closely the title matches its wording.\n\n"
            f"CLAIM: {gap.claim}\n"
            f"TYPE: {gap.claim_type.value}\n"
            f"PREFERRED SOURCE TYPE: {gap.preferred_source_type}\n\n"
            "Source tiers, best first:\n"
            "  primary            the work that first established this\n"
            "  original           the paper introducing the method concerned\n"
            "  official_benchmark the benchmark's own publication\n"
            "  authoritative      a survey or standard reference by recognised authors\n"
            "  secondary          anything else that reliably reports it\n\n"
            "A later paper that merely mentions the claim is NOT a primary source. "
            "Set best_index to -1 if none of these would actually settle the claim.\n\n"
            f"CANDIDATES:\n{listing[:30000]}"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=RANK_SCHEMA, temperature=0.1)
        data = result["data"]

        out: list[CitationCandidate] = []
        for r in sorted(data.get("ranked", []),
                        key=lambda x: -float(x.get("supports_claim_likelihood", 0))):
            idx = int(r.get("index", -1))
            if not (0 <= idx < len(candidates)):
                continue
            c = candidates[idx]
            out.append(CitationCandidate(
                title=c.get("title", ""),
                authors=[str(a) for a in (c.get("authors") or [])],
                year=_year(c),
                venue=c.get("venue", "") or "",
                doi=c.get("doi", "") or "",
                arxiv_id=c.get("arxiv_id", "") or "",
                url=c.get("url", "") or "",
                abstract=(c.get("abstract") or "")[:4000],
                source=SourceRef(
                    source=str(c.get("source", "openalex")),
                    source_id=str(c.get("source_id", c.get("doi", ""))),
                    url=str(c.get("url", "")), title=str(c.get("title", ""))),
                relevance_rationale=r.get("rationale", ""),
                source_tier=r.get("source_tier", "secondary"),
            ))
        return out


class CitationVerificationSkill(Skill):
    """Read the source and decide whether it really supports the claim."""

    name = "citation_verification"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("fetch_openalex_work", "fetch_arxiv_paper", "download_pdf",
                      "parse_pdf", "verify_doi", "resolve_paper_identity", "fetch_url")

    async def run(
        self, ctx: AgentContext, *, claim: str, candidate: CitationCandidate,
        claim_type: ClaimType = ClaimType.DEFINITIONAL, deep: bool = True, **_: Any,
    ) -> CitationVerification:
        cfg = get_config()

        identity = await ctx.try_call(
            "resolve_paper_identity", default=None,
            title=candidate.title, doi=candidate.doi, arxiv_id=candidate.arxiv_id)

        full_text = candidate.abstract
        source_of_text = "abstract"
        # A numerical or benchmark claim cannot be verified from an abstract: the
        # setup and the exact numbers live in the body.
        needs_body = claim_type.value in cfg.get(
            "citation.require_primary_source_for", ["numerical", "benchmark", "sota", "causal"])
        if deep and needs_body and "parse_pdf" in ctx.allowed_tools:
            pdf_url = ""
            if candidate.arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{candidate.arxiv_id}"
            elif identity and identity.get("url", "").endswith(".pdf"):
                pdf_url = identity["url"]
            if pdf_url:
                parsed = await ctx.try_call("parse_pdf", default=None, url=pdf_url)
                if parsed and parsed.get("text"):
                    sections = parsed.get("sections", {})
                    picked = " ".join(
                        sections.get(k, "")[:8000] for k in
                        ("abstract", "results", "evaluation", "experiment",
                         "experiments", "analysis", "limitations", "conclusion")
                        if sections.get(k))
                    full_text = (picked or parsed["text"])[:40000]
                    source_of_text = f"full text ({parsed['pages_parsed']} pages)"

        identity_block = ""
        if identity and identity.get("resolved"):
            identity_block = (
                f"AUTHORITATIVE METADATA (resolved via {identity.get('sources_consulted')}):\n"
                f"  title: {identity.get('title')}\n"
                f"  authors: {', '.join(identity.get('authors', [])[:8])}\n"
                f"  year: {identity.get('year')}  venue: {identity.get('venue')}\n"
                f"  doi: {identity.get('doi')}  arxiv: {identity.get('arxiv_id')}\n"
                f"  title agreement with candidate: {identity.get('title_agreement')}\n\n"
            )

        prompt = (
            "Verify whether this source actually supports the claim a textbook wants "
            "to attach to it. Be sceptical: your job is to reject bad citations, not "
            "to find a way to accept them.\n\n"
            f"CLAIM THE BOOK WANTS TO MAKE:\n  {claim}\n"
            f"CLAIM TYPE: {claim_type.value}\n\n"
            f"{identity_block}"
            f"CANDIDATE SOURCE ({source_of_text}):\n"
            f"  title: {candidate.title}\n"
            f"  authors: {', '.join(candidate.authors[:8])}\n"
            f"  year: {candidate.year}  venue: {candidate.venue}\n"
            f"  doi: {candidate.doi}  arxiv: {candidate.arxiv_id}\n\n"
            f"SOURCE CONTENT:\n{full_text[:40000]}\n\n"
            "=== CHECK, IN ORDER ===\n"
            "1. IDENTITY. Do the title, authors, year and venue match? A mismatch means "
            "the citation points at a different work.\n"
            "2. SUPPORT. Does this source ESTABLISH the claim, or merely mention it? "
            "Quote the exact passage that supports it. If you cannot quote one, it does "
            "not support the claim.\n"
            "3. NUMBERS. If the claim is numeric, do the reported numbers match, under "
            "what experimental setup, and on what benchmark?\n"
            "4. CITATION LAUNDERING. This is the critical check. If this source says "
            "something like 'Smith et al. showed X' or 'prior work reports X', then it "
            "REPORTS the claim rather than establishing it. Set laundering_detected=true, "
            "name the real primary source, and return verdict=needs_primary — even if "
            "the claim is true. A textbook citing a paper that cites another paper is "
            "not evidence.\n"
            "5. CAVEATS. What conditions must the book state for the claim to be honest?\n\n"
            "Return verdict=accept ONLY when identity is confirmed, the source "
            "establishes the claim, and you quoted the supporting passage."
        )

        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=VERIFY_SCHEMA, temperature=0.05)
        d = result["data"]

        verdict = d.get("verdict", "reject")
        laundered = bool(d.get("laundering_detected"))
        if laundered and cfg.get("citation.reject_on_laundering", True):
            verdict = "needs_primary"
        if not d.get("identity_confirmed"):
            verdict = "reject"
        if d.get("supports_claim") in ("does_not_support", "unverifiable"):
            verdict = "reject"
        # A numeric claim accepted without a quoted passage is exactly the failure
        # mode this subsystem exists to prevent.
        if (verdict == "accept" and needs_body
                and not (d.get("evidence_quote") or "").strip()):
            verdict = "reject"
            d["note"] = ((d.get("note") or "") +
                         " | rejected: no verbatim passage quoted for a claim "
                         "requiring primary evidence").strip(" |")

        return CitationVerification(
            candidate=candidate,
            identity_confirmed=bool(d.get("identity_confirmed")),
            supports_claim=d.get("supports_claim", "unverifiable"),
            evidence_quote=(d.get("evidence_quote") or "")[:2000],
            experimental_setup=(d.get("experimental_setup") or "")[:1500],
            reported_numbers=(d.get("reported_numbers") or "")[:800],
            caveats=(d.get("caveats") or [])[:6],
            laundering_detected=laundered,
            laundering_note=(d.get("laundering_note") or "")[:1000],
            true_primary_source=(d.get("true_primary_source") or "")[:400],
            verdict=verdict,
            note=(d.get("note") or "")[:600],
        )


# -- helpers ---------------------------------------------------------------


def _dedupe_candidates(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in raw:
        if not c or not c.get("title"):
            continue
        key = (c.get("doi") or c.get("arxiv_id")
               or re.sub(r"[^a-z0-9]", "", str(c.get("title", "")).lower())[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    out.sort(key=lambda c: -(c.get("citation_count") or 0))
    return out


def _year(c: dict[str, Any]) -> int | None:
    if c.get("year"):
        try:
            return int(c["year"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d{4})", str(c.get("published", "") or c.get("published_at", "")))
    return int(m.group(1)) if m else None


def _claim_type(value: Any) -> ClaimType:
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.DEFINITIONAL
