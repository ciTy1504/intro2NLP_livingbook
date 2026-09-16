"""Claim extraction.

Shared deliberately: the same contract is used to extract the claims a *paper* makes
and the claims the *book* makes. Both feed the citation subsystem, which does not care
where a claim came from — only whether it is supported.

The distinction that matters here is between a claim and a statement. "The Transformer
uses self-attention" is a definition; "GQA matches MHA quality with 8x less KV cache"
is a claim, and a textbook owes the reader a source for the second.
"""

from __future__ import annotations

from typing import Any

from ...research.models import Claim, ClaimType, SourceRef
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, json_schema, number, obj, string

CLAIM_SCHEMA = json_schema(
    {
        "claims": array_of(obj({
            "text": string(
                "The claim in English, self-contained — a reader must be able to "
                "check it without the surrounding text"),
            "claim_type": string("", ["numerical", "benchmark", "historical", "causal",
                                      "sota", "definitional", "architectural",
                                      "attribution"]),
            "needs_citation": boolean("Does this require a source it does not have?"),
            "is_author_claim": boolean(
                "True if the source asserts this about its own work without "
                "independent verification"),
            "existing_citation": string("Citation key already attached, if any"),
            "verbatim": string("The original sentence this came from"),
            "confidence": number("0..1, how confident you are this IS a claim"),
        }, ["text", "claim_type", "needs_citation"])),
        "non_claims_noted": array_of(string(),
                                     "Statements deliberately not treated as claims"),
    },
    ["claims"],
)


class ClaimExtractionSkill(Skill):
    name = "claim_extraction"
    required_tools = ("gemini_structured_output",)

    async def run(
        self,
        ctx: AgentContext,
        *,
        text: str,
        context: str = "",
        source: SourceRef | None = None,
        language: str = "en",
        max_claims: int = 25,
        **_: Any,
    ) -> list[Claim]:
        prompt = (
            "Extract the checkable factual claims from this text.\n\n"
            + ("The text is in Vietnamese; answer in ENGLISH.\n\n"
               if language == "vi" else "")
            + "A CLAIM is a statement a reader could verify against a source:\n"
              "  numerical      a specific measurement, size, cost or speed\n"
              "  benchmark      a score on a named benchmark\n"
              "  sota           an assertion of being best/fastest/largest\n"
              "  historical     who did what, and when\n"
              "  causal         X causes or enables Y\n"
              "  architectural  a model or system works in a particular way\n"
              "  attribution    a method is due to particular authors\n"
              "  definitional   a term means a particular thing\n\n"
            "NOT claims: pedagogical framing, worked examples with invented numbers, "
            "statements about the document's own structure, opinions, and general "
            "knowledge a technical reader already shares.\n\n"
            "Set needs_citation=false only when the text already carries an adequate "
            "citation, or the claim is definitional and uncontested.\n"
            "Set is_author_claim=true when the source is asserting something about its "
            "own work that nobody else has verified — that distinction decides how much "
            "weight the claim can carry.\n\n"
            + (f"CONTEXT: {context}\n\n" if context else "")
            + f"TEXT:\n{text[:30000]}"
        )

        result = await ctx.call(
            "gemini_structured_output", prompt=prompt, schema=CLAIM_SCHEMA,
            temperature=0.1)

        claims: list[Claim] = []
        for c in (result["data"].get("claims") or [])[:max_claims]:
            if not (c.get("text") or "").strip():
                continue
            claims.append(Claim(
                text=c["text"],
                claim_type=_claim_type(c.get("claim_type")),
                needs_citation=bool(c.get("needs_citation", True)),
                confidence=float(c.get("confidence", 0.6) or 0.6),
                supporting_sources=[source] if source else [],
            ))
        return claims


def _claim_type(value: Any) -> ClaimType:
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.DEFINITIONAL
