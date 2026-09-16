"""BibTeX verification and safe writing.

Verification is independent of whatever produced the entry: fields are checked against
metadata resolved from DOI registries, not against the model's recollection of the
paper. This is the last gate before a citation becomes part of the book, and it is the
only component permitted to write ``references.bib``.
"""

from __future__ import annotations

import re
from typing import Any

from ...config import get_config
from ...knowledge.bib import Bibliography, bibtex_from_metadata
from ...research.models import CitationCandidate
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, json_schema, obj, string

FIELD_CHECK_SCHEMA = json_schema(
    {
        "matches": boolean("Do the BibTeX fields agree with the authoritative metadata?"),
        "problems": array_of(obj({
            "field": string(),
            "bibtex_value": string(),
            "authoritative_value": string(),
            "severity": string("", ["blocker", "major", "minor"]),
        }, ["field", "severity"])),
        "corrected": obj({
            "title": string(), "author": string(), "year": string(), "venue": string(),
            "doi": string(), "eprint": string(), "url": string(),
            "entry_type": string("", ["article", "inproceedings", "book", "incollection",
                                      "techreport", "misc", "phdthesis"]),
        }),
        "note": string(),
    },
    ["matches", "problems"],
)


class BibtexVerificationSkill(Skill):
    name = "bibtex_verification"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("retrieve_bibtex", "verify_doi", "bib_validate", "read_file",
                      "write_file")

    async def run(
        self, ctx: AgentContext, *, candidate: CitationCandidate | None = None,
        bib_key: str = "", write: bool = False, **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        bib = Bibliography(cfg.bib_path)

        if bib_key and not candidate:
            return await self._verify_existing(ctx, bib, bib_key)
        if candidate:
            return await self._prepare_new(ctx, bib, candidate, write=write)
        raise ValueError("bibtex_verification needs a candidate or a bib_key")

    # -- existing entries --------------------------------------------------
    async def _verify_existing(
        self, ctx: AgentContext, bib: Bibliography, bib_key: str,
    ) -> dict[str, Any]:
        entry = bib.get(bib_key)
        if not entry:
            return {"bib_key": bib_key, "exists": False,
                    "verdict": "missing", "problems": []}

        authoritative = None
        if entry.doi and "verify_doi" in ctx.allowed_tools:
            authoritative = await ctx.try_call("verify_doi", default=None, doi=entry.doi)
        if not authoritative and entry.arxiv_id and "fetch_arxiv_paper" in ctx.allowed_tools:
            authoritative = await ctx.try_call(
                "fetch_arxiv_paper", default=None, arxiv_id=entry.arxiv_id)

        if not authoritative:
            # No resolvable identifier is a real weakness in a textbook bibliography,
            # but it is not evidence the entry is wrong.
            return {
                "bib_key": bib_key, "exists": True, "verdict": "unverifiable",
                "reason": "no DOI or arXiv id to resolve against",
                "problems": [], "entry": entry.to_row(),
            }

        prompt = (
            "Compare a BibTeX entry against authoritative metadata resolved from a DOI "
            "or arXiv registry, and report disagreements.\n\n"
            "Treat as MINOR: abbreviated vs full venue names, initials vs full given "
            "names, author order preserved but formatted differently, missing page "
            "numbers, preprint year vs publication year differing by one.\n"
            "Treat as BLOCKER: a different paper entirely, wrong authors, a year off by "
            "more than one, a DOI pointing elsewhere.\n\n"
            f"BIBTEX ENTRY ({bib_key}):\n"
            f"  title: {entry.title}\n  author: {entry.authors}\n  year: {entry.year}\n"
            f"  venue: {entry.venue}\n  doi: {entry.doi}\n  arxiv: {entry.arxiv_id}\n"
            f"  type: {entry.entry_type}\n\n"
            f"AUTHORITATIVE METADATA:\n"
            f"  title: {authoritative.get('title')}\n"
            f"  authors: {', '.join(authoritative.get('authors', [])[:10])}\n"
            f"  year: {authoritative.get('year')}\n"
            f"  venue: {authoritative.get('venue')}\n"
            f"  doi: {authoritative.get('doi')}\n"
            f"  type: {authoritative.get('type')}\n"
        )
        result = await ctx.call("gemini_structured_output", prompt=prompt,
                                schema=FIELD_CHECK_SCHEMA, temperature=0.05, role="fast")
        d = result["data"]
        blockers = [p for p in d.get("problems", []) if p.get("severity") == "blocker"]
        return {
            "bib_key": bib_key, "exists": True,
            "verdict": "invalid" if blockers else ("suspect" if d.get("problems") else "valid"),
            "problems": d.get("problems", []),
            "corrected": d.get("corrected", {}),
            "note": d.get("note", ""),
            "entry": entry.to_row(),
        }

    # -- new entries -------------------------------------------------------
    async def _prepare_new(
        self, ctx: AgentContext, bib: Bibliography, candidate: CitationCandidate,
        *, write: bool,
    ) -> dict[str, Any]:
        """Build a verified entry from authoritative metadata, reusing an existing key
        where the source is already in the bibliography."""
        existing = _find_existing(bib, candidate)
        if existing:
            return {"bib_key": existing, "created": False, "reused": True,
                    "verdict": "valid",
                    "note": f"source already in the bibliography as {existing}"}

        raw_bibtex = ""
        if "retrieve_bibtex" in ctx.allowed_tools:
            raw_bibtex = await ctx.try_call(
                "retrieve_bibtex", default="",
                doi=candidate.doi, arxiv_id=candidate.arxiv_id) or ""

        authoritative: dict[str, Any] = {}
        if candidate.doi and "verify_doi" in ctx.allowed_tools:
            authoritative = await ctx.try_call(
                "verify_doi", default=None, doi=candidate.doi) or {}
        if not authoritative and candidate.arxiv_id and "fetch_arxiv_paper" in ctx.allowed_tools:
            authoritative = await ctx.try_call(
                "fetch_arxiv_paper", default=None, arxiv_id=candidate.arxiv_id) or {}

        if not authoritative and not raw_bibtex:
            # Refusing here is the point: an entry assembled from a model's memory of
            # a paper is precisely the kind of citation this system must not produce.
            return {"bib_key": "", "created": False, "verdict": "unverifiable",
                    "note": "no authoritative metadata could be resolved; entry refused"}

        title = authoritative.get("title") or candidate.title
        authors = authoritative.get("authors") or candidate.authors
        year = authoritative.get("year") or candidate.year
        venue = authoritative.get("venue") or candidate.venue
        doi = authoritative.get("doi") or candidate.doi
        arxiv_id = authoritative.get("arxiv_id") or candidate.arxiv_id
        url = authoritative.get("url") or candidate.url

        entry_type, fields = bibtex_from_metadata(
            title=title, authors=authors, year=year, venue=venue,
            doi=doi, arxiv_id=arxiv_id, url=url)

        key = bib.suggest_key(fields.get("author", ""), year, title)
        rendered = bib.render_entry(key, entry_type, fields)

        created = False
        if write and "write_file" in ctx.allowed_tools:
            # Appended through the Bibliography object so the file's hand-maintained
            # chapter banners and formatting survive; the file is never regenerated.
            ctx.check_write_path(str(get_config().bib_path))
            bib.append(key, entry_type, fields,
                       comment=f"added by livingbook: {candidate.relevance_rationale[:120]}")
            created = True

        return {
            "bib_key": key, "created": created, "reused": False, "verdict": "valid",
            "entry_type": entry_type, "fields": fields, "bibtex": rendered,
            "retrieved_bibtex": raw_bibtex[:2000],
            "metadata_source": authoritative.get("resolver") or authoritative.get("source", ""),
        }


def _find_existing(bib: Bibliography, candidate: CitationCandidate) -> str:
    """Reuse rather than duplicate: match on DOI, arXiv id, then normalised title."""
    if candidate.doi:
        for key, entry in bib.entries.items():
            if entry.doi and entry.doi.lower() == candidate.doi.lower():
                return key
    if candidate.arxiv_id:
        for key, entry in bib.entries.items():
            if entry.arxiv_id == candidate.arxiv_id:
                return key
    norm = re.sub(r"[^a-z0-9]", "", candidate.title.lower())
    if len(norm) > 18:
        for key, entry in bib.entries.items():
            if re.sub(r"[^a-z0-9]", "", entry.title.lower()) == norm:
                return key
    return ""
