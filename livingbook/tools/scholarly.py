"""Scholarly source tools: arXiv, OpenAlex, Crossref, DataCite, Semantic Scholar,
OpenReview, ACL Anthology, PDF retrieval and parsing.

Source selection reflects what was measured, not what is best known (AUDIT.md §6):
OpenAlex is primary because it answers reliably without a key, Semantic Scholar is
optional because it returns 429 unauthenticated, and DOI verification routes by prefix
because Crossref 404s on arXiv's ``10.48550/*`` DOIs while DataCite resolves them.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import get_config, get_secrets
from ..obs import get_logger
from .http import get_bytes, get_json, get_text, polite_params, request
from .registry import Capability, ToolError, ToolUnavailable, tool

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org"
CROSSREF_API = "https://api.crossref.org"
DATACITE_API = "https://api.datacite.org"
S2_API = "https://api.semanticscholar.org/graph/v1"
OPENREVIEW_API = "https://api2.openreview.net"
ACL_BASE = "https://aclanthology.org"

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


# ───────────────────────────────── arXiv ─────────────────────────────────


@tool("search_arxiv", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Search arXiv by query, category and date window.")
async def search_arxiv(
    query: str = "",
    categories: list[str] | None = None,
    *,
    max_results: int = 25,
    since_days: int | None = None,
    sort_by: str = "submittedDate",
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    if query:
        escaped = query.replace('"', "")
        clauses.append(f'(abs:"{escaped}" OR ti:"{escaped}")')
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        clauses.append(f"({cats})")
    search_query = " AND ".join(clauses) if clauses else "cat:cs.CL"

    text = await get_text(
        ARXIV_API,
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": "descending",
        },
        timeout=60,
    )
    papers = _parse_arxiv_atom(text)

    if since_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        papers = [p for p in papers if _parse_dt(p.get("published")) >= cutoff]
    return papers


@tool("fetch_arxiv_paper", [Capability.FETCH, Capability.SCHOLARLY],
      description="Fetch full arXiv metadata for one paper id.")
async def fetch_arxiv_paper(arxiv_id: str) -> dict[str, Any] | None:
    clean = _normalise_arxiv_id(arxiv_id)
    text = await get_text(ARXIV_API, params={"id_list": clean, "max_results": 1}, timeout=45)
    papers = _parse_arxiv_atom(text)
    return papers[0] if papers else None


def _parse_arxiv_atom(text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ToolError(f"arXiv returned unparseable Atom: {exc}") from exc

    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = _text(entry, "a:id")
        m = _ARXIV_ID_RE.search(raw_id or "")
        arxiv_id = m.group(1) if m else (raw_id or "").rsplit("/", 1)[-1]

        pdf_url = ""
        for link in entry.findall("a:link", _ATOM):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")

        out.append({
            "source": "arxiv",
            "source_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": _clean(_text(entry, "a:title")),
            "abstract": _clean(_text(entry, "a:summary")),
            "authors": [
                _clean(_text(a, "a:name")) for a in entry.findall("a:author", _ATOM)
            ],
            "published": _text(entry, "a:published"),
            "updated": _text(entry, "a:updated"),
            "categories": [c.get("term") for c in entry.findall("a:category", _ATOM)],
            "primary_category": (
                entry.find("arxiv:primary_category", _ATOM).get("term")
                if entry.find("arxiv:primary_category", _ATOM) is not None else None
            ),
            "doi": _text(entry, "arxiv:doi") or f"10.48550/arXiv.{arxiv_id}",
            "journal_ref": _text(entry, "arxiv:journal_ref"),
            "comment": _text(entry, "arxiv:comment"),
            "url": raw_id,
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return out


# ──────────────────────────────── OpenAlex ───────────────────────────────


@tool("search_openalex", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Search OpenAlex works; the primary scholarly backend.")
async def search_openalex(
    query: str,
    *,
    per_page: int = 25,
    since_days: int | None = None,
    min_citations: int = 0,
    concept: str = "",
) -> list[dict[str, Any]]:
    filters: list[str] = []
    if since_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).date()
        filters.append(f"from_publication_date:{cutoff.isoformat()}")
    if min_citations:
        filters.append(f"cited_by_count:>{min_citations - 1}")
    if concept:
        filters.append(f"concepts.display_name.search:{concept}")

    params = polite_params({
        "search": query,
        "per-page": min(per_page, 50),
        "sort": "publication_date:desc" if since_days else "relevance_score:desc",
    })
    if filters:
        params["filter"] = ",".join(filters)

    data = await get_json(f"{OPENALEX_API}/works", params=params, timeout=60)
    return [_normalise_openalex(w) for w in data.get("results", [])]


@tool("fetch_openalex_work", [Capability.FETCH, Capability.SCHOLARLY],
      description="Fetch one OpenAlex work by id, DOI or arXiv id.")
async def fetch_openalex_work(identifier: str) -> dict[str, Any] | None:
    ident = identifier.strip()
    if ident.lower().startswith("10."):
        url = f"{OPENALEX_API}/works/doi:{ident}"
    elif _ARXIV_ID_RE.fullmatch(ident):
        url = f"{OPENALEX_API}/works/doi:10.48550/arXiv.{ident}"
    elif ident.startswith("http"):
        url = ident if "openalex.org" in ident else f"{OPENALEX_API}/works/doi:{ident}"
    else:
        url = f"{OPENALEX_API}/works/{ident}"
    try:
        return _normalise_openalex(await get_json(url, params=polite_params(), timeout=45))
    except ToolError:
        return None


def _normalise_openalex(work: dict[str, Any]) -> dict[str, Any]:
    ids = work.get("ids", {}) or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "")
    primary = work.get("primary_location") or {}
    venue = ((primary.get("source") or {}).get("display_name")) or ""
    arxiv_id = ""
    m = _ARXIV_ID_RE.search(doi or "")
    if "arxiv" in doi.lower() and m:
        arxiv_id = m.group(1)

    return {
        "source": "openalex",
        "source_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "title": work.get("display_name") or work.get("title") or "",
        "abstract": _invert_abstract(work.get("abstract_inverted_index")),
        "authors": [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
        ],
        "year": work.get("publication_year"),
        "published": work.get("publication_date"),
        "venue": venue,
        "venue_type": primary.get("version"),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "citation_count": work.get("cited_by_count", 0),
        "is_open_access": (work.get("open_access") or {}).get("is_oa", False),
        "pdf_url": (work.get("best_oa_location") or {}).get("pdf_url") or "",
        "concepts": [
            c.get("display_name") for c in (work.get("concepts") or [])[:8]
            if c.get("score", 0) > 0.3
        ],
        "referenced_works": work.get("referenced_works", [])[:50],
        "url": ids.get("doi") or work.get("id") or "",
        "type": work.get("type"),
    }


def _invert_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index for licensing reasons."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    return " ".join(w for _, w in sorted(positions))[:6000]


# ──────────────────────── Crossref / DataCite / DOI ──────────────────────


@tool("search_crossref", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Bibliographic search against Crossref.")
async def search_crossref(query: str, *, rows: int = 10) -> list[dict[str, Any]]:
    data = await get_json(
        f"{CROSSREF_API}/works",
        params=polite_params({"query.bibliographic": query, "rows": min(rows, 30)}),
        timeout=60,
    )
    return [_normalise_crossref(it) for it in data.get("message", {}).get("items", [])]


@tool("verify_doi", [Capability.FETCH, Capability.SCHOLARLY],
      description="Resolve a DOI to authoritative metadata via Crossref or DataCite.")
async def verify_doi(doi: str) -> dict[str, Any] | None:
    """Resolve a DOI, routing by registrant prefix.

    arXiv DOIs (``10.48550/*``) are registered with DataCite, not Crossref; sending
    them to Crossref returns 404 and would make every arXiv citation look invalid.
    """
    clean = doi.strip().replace("https://doi.org/", "").replace("doi:", "")
    if not clean:
        return None

    cfg = get_config()
    resolvers = cfg.get("citation.doi_resolvers", {}) or {}
    prefix = clean.split("/")[0]
    resolver = resolvers.get(prefix, resolvers.get("default", "crossref"))

    order = ["datacite", "crossref"] if resolver == "datacite" else ["crossref", "datacite"]
    for backend in order:
        try:
            if backend == "crossref":
                data = await get_json(f"{CROSSREF_API}/works/{clean}",
                                      params=polite_params(), timeout=45)
                return {**_normalise_crossref(data["message"]), "resolver": "crossref",
                        "resolved": True}
            data = await get_json(f"{DATACITE_API}/dois/{clean}", timeout=45)
            return {**_normalise_datacite(data["data"]), "resolver": "datacite",
                    "resolved": True}
        except (ToolError, ToolUnavailable, KeyError):
            continue
    return None


def _normalise_crossref(item: dict[str, Any]) -> dict[str, Any]:
    parts = (item.get("published-print") or item.get("published-online")
             or item.get("issued") or {}).get("date-parts", [[None]])
    year = parts[0][0] if parts and parts[0] else None
    return {
        "source": "crossref",
        "source_id": item.get("DOI", ""),
        "title": (item.get("title") or [""])[0],
        "authors": [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in (item.get("author") or [])
        ],
        "year": year,
        "venue": (item.get("container-title") or [""])[0],
        "doi": item.get("DOI", ""),
        "type": item.get("type"),
        "publisher": item.get("publisher"),
        "url": item.get("URL", ""),
        "citation_count": item.get("is-referenced-by-count", 0),
        "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract", "") or "").strip(),
    }


def _normalise_datacite(data: dict[str, Any]) -> dict[str, Any]:
    attrs = data.get("attributes", {}) or {}
    titles = attrs.get("titles") or [{}]
    descriptions = attrs.get("descriptions") or []
    return {
        "source": "datacite",
        "source_id": attrs.get("doi", ""),
        "title": titles[0].get("title", ""),
        "authors": [
            c.get("name") or " ".join(filter(None, [c.get("givenName"), c.get("familyName")]))
            for c in (attrs.get("creators") or [])
        ],
        "year": attrs.get("publicationYear"),
        "venue": attrs.get("publisher") if isinstance(attrs.get("publisher"), str)
                 else (attrs.get("publisher") or {}).get("name", ""),
        "doi": attrs.get("doi", ""),
        "type": (attrs.get("types") or {}).get("resourceTypeGeneral"),
        "url": attrs.get("url", ""),
        "abstract": next(
            (d.get("description", "") for d in descriptions
             if d.get("descriptionType") == "Abstract"), ""),
    }


@tool("resolve_paper_identity", [Capability.FETCH, Capability.SCHOLARLY],
      description="Resolve a paper across arXiv/OpenAlex/Crossref into one identity.")
async def resolve_paper_identity(
    *, title: str = "", doi: str = "", arxiv_id: str = "",
) -> dict[str, Any]:
    """Establish what a paper actually is, from whatever partial reference we hold.

    The citation verifier needs this: a `\\cite` key, a title from a model, and a URL
    from a blog post may all refer to the same work, and deciding whether a citation
    supports a claim requires knowing which work is meant.
    """
    candidates: list[dict[str, Any]] = []

    if doi:
        resolved = await verify_doi(doi)
        if resolved:
            candidates.append(resolved)
    if arxiv_id:
        paper = await fetch_arxiv_paper(arxiv_id)
        if paper:
            candidates.append(paper)
    if not candidates and title:
        try:
            oa = await search_openalex(title, per_page=3)
            candidates.extend(oa[:2])
        except (ToolError, ToolUnavailable):
            pass
        if not candidates:
            try:
                candidates.extend((await search_crossref(title, rows=2))[:1])
            except (ToolError, ToolUnavailable):
                pass

    if not candidates:
        return {"resolved": False, "query": {"title": title, "doi": doi, "arxiv_id": arxiv_id}}

    best = candidates[0]
    agreement = None
    if title and best.get("title"):
        agreement = _title_similarity(title, best["title"])

    return {
        "resolved": True,
        "title": best.get("title", ""),
        "authors": best.get("authors", []),
        "year": best.get("year") or _year_from(best.get("published")),
        "venue": best.get("venue", ""),
        "doi": best.get("doi", ""),
        "arxiv_id": best.get("arxiv_id", ""),
        "url": best.get("url", ""),
        "citation_count": best.get("citation_count"),
        "abstract": best.get("abstract", "")[:4000],
        "title_agreement": agreement,
        "sources_consulted": [c.get("source") for c in candidates],
    }


@tool("retrieve_bibtex", [Capability.FETCH, Capability.SCHOLARLY],
      description="Retrieve authoritative BibTeX for a DOI or arXiv id.")
async def retrieve_bibtex(*, doi: str = "", arxiv_id: str = "") -> str:
    if doi:
        try:
            resp = await request(
                "GET", f"https://doi.org/{doi.replace('https://doi.org/', '')}",
                headers={"Accept": "application/x-bibtex"}, timeout=45,
            )
            if resp.status_code == 200 and "@" in resp.text:
                return resp.text.strip()
        except (ToolError, ToolUnavailable):
            pass
    if arxiv_id:
        paper = await fetch_arxiv_paper(arxiv_id)
        if paper:
            authors = " and ".join(paper["authors"])
            year = _year_from(paper.get("published")) or ""
            return (
                f"@article{{arxiv{arxiv_id.replace('.', '')},\n"
                f"  title={{{paper['title']}}},\n"
                f"  author={{{authors}}},\n"
                f"  journal={{arXiv preprint arXiv:{arxiv_id}}},\n"
                f"  year={{{year}}},\n"
                f"  eprint={{{arxiv_id}}},\n"
                f"  archivePrefix={{arXiv}},\n"
                f"  url={{https://arxiv.org/abs/{arxiv_id}}}\n}}"
            )
    raise ToolError("no BibTeX could be retrieved for the given identifiers")


# ────────────────────────── Semantic Scholar ─────────────────────────────


@tool("search_semantic_scholar", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Search Semantic Scholar (optional; requires a key in practice).")
async def search_semantic_scholar(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Measured 429 without a key, so this is disabled unless one is configured.

    Raising ToolUnavailable rather than returning [] is deliberate: callers use
    ``try_call`` and degrade, and a silent empty list would look like "no results".
    """
    secrets = get_secrets()
    if not secrets.has("SEMANTIC_SCHOLAR_API_KEY"):
        if not get_config().get("research.sources.semantic_scholar.enabled", False):
            raise ToolUnavailable(
                "Semantic Scholar is disabled: it rate-limits unauthenticated clients. "
                "Set SEMANTIC_SCHOLAR_API_KEY in secrets/.env to enable."
            )
    headers = {}
    if secrets.has("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = secrets.require("SEMANTIC_SCHOLAR_API_KEY")

    fields = "title,abstract,year,venue,citationCount,externalIds,authors,openAccessPdf"
    data = await get_json(
        f"{S2_API}/paper/search",
        params={"query": query, "limit": min(limit, 20), "fields": fields},
        headers=headers, timeout=45,
    )
    out = []
    for p in data.get("data", []) or []:
        ext = p.get("externalIds") or {}
        out.append({
            "source": "semantic_scholar",
            "source_id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract") or "",
            "authors": [a.get("name", "") for a in (p.get("authors") or [])],
            "year": p.get("year"),
            "venue": p.get("venue", ""),
            "doi": ext.get("DOI", ""),
            "arxiv_id": ext.get("ArXiv", ""),
            "citation_count": p.get("citationCount", 0),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            "url": f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
        })
    return out


# ───────────────────────── OpenReview / ACL ──────────────────────────────


@tool("search_openreview", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Search OpenReview for submissions, acceptances and reviews.")
async def search_openreview(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    data = await get_json(
        f"{OPENREVIEW_API}/notes/search",
        params={"query": query, "limit": min(limit, 50)}, timeout=60,
    )
    out = []
    for note in data.get("notes", []) or []:
        content = note.get("content", {}) or {}

        def field(name: str) -> str:
            value = content.get(name)
            if isinstance(value, dict):
                value = value.get("value")
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value) if value is not None else ""

        out.append({
            "source": "openreview",
            "source_id": note.get("id", ""),
            "title": field("title"),
            "abstract": field("abstract"),
            "authors": [a for a in field("authors").split(", ") if a],
            "venue": field("venue") or field("venueid"),
            "keywords": [k for k in field("keywords").split(", ") if k],
            "pdf_url": f"https://openreview.net/pdf?id={note.get('id','')}",
            "url": f"https://openreview.net/forum?id={note.get('id','')}",
            "published": _ms_to_iso(note.get("cdate")),
            "decision": field("decision"),
        })
    return out


@tool("search_acl_anthology", [Capability.SEARCH, Capability.SCHOLARLY],
      description="Search the ACL Anthology via OpenAlex venue filtering.")
async def search_acl_anthology(query: str, *, limit: int = 15) -> list[dict[str, Any]]:
    """ACL Anthology has no search API, so this queries OpenAlex restricted to *ACL venues.

    The full 12.6 MB anthology.bib dump is available and verified reachable, but
    downloading it per query is wasteful; it is better suited to a periodic bulk
    refresh than to interactive search.
    """
    results = await search_openalex(query, per_page=limit * 2)
    acl_venues = ("acl", "emnlp", "naacl", "eacl", "coling", "conll",
                  "computational linguistics", "transactions of the association")
    filtered = [
        r for r in results
        if any(v in (r.get("venue") or "").lower() for v in acl_venues)
    ]
    for r in filtered:
        r["source"] = "acl"
    return filtered[:limit]


# ─────────────────────────────── PDFs ────────────────────────────────────


@tool("download_pdf", [Capability.FETCH, Capability.SCHOLARLY],
      description="Download a PDF and return its bytes.")
async def download_pdf(url: str, *, max_mb: int = 40) -> bytes:
    data = await get_bytes(url, max_bytes=max_mb * 1024 * 1024, timeout=120)
    if not data.startswith(b"%PDF"):
        raise ToolError(f"{url} did not return a PDF (got {data[:16]!r})")
    return data


@tool("parse_pdf", [Capability.FETCH],
      description="Extract text and section structure from a PDF.")
async def parse_pdf(
    pdf_bytes: bytes | None = None, *, url: str = "", max_pages: int = 40,
    max_chars: int = 120_000,
) -> dict[str, Any]:
    """Extract text from a paper PDF.

    Used by the technical verifier and citation verifier, which must check a claim
    against what a paper actually reports rather than against an abstract or an LLM's
    recollection of it.
    """
    if pdf_bytes is None:
        if not url:
            raise ToolError("parse_pdf needs either pdf_bytes or url")
        pdf_bytes = await download_pdf(url)

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ToolUnavailable("PyMuPDF is not installed; cannot parse PDFs") from exc

    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as exc:
        raise ToolError(f"could not open PDF: {exc}") from exc

    pages: list[str] = []
    total = 0
    try:
        for i, page in enumerate(doc):
            if i >= max_pages or total >= max_chars:
                break
            text = page.get_text("text")
            pages.append(text)
            total += len(text)
        meta = dict(doc.metadata or {})
        n_pages = doc.page_count
    finally:
        doc.close()

    full = "\n".join(pages)[:max_chars]
    return {
        "text": full,
        "pages_parsed": len(pages),
        "total_pages": n_pages,
        "chars": len(full),
        "metadata": {k: v for k, v in meta.items() if v},
        "sections": _pdf_sections(full),
        "truncated": n_pages > len(pages),
    }


_SECTION_HEAD = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*)\s+)?"
    r"(abstract|introduction|related work|background|method(?:s|ology)?|approach|"
    r"experiment(?:s|al setup)?|results?|evaluation|analysis|ablation|discussion|"
    r"limitations?|conclusions?|references|appendix)\b.*$",
    re.I | re.M,
)


def _pdf_sections(text: str) -> dict[str, str]:
    """Split paper text on conventional section headings.

    Crude, but it lets a verifier read "the Limitations section" instead of the whole
    paper — which is both cheaper and more likely to answer the question.
    """
    matches = list(_SECTION_HEAD.finditer(text))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(2).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body and name not in out:
            out[name] = body[:20000]
    return out


# ─────────────────────────────── helpers ─────────────────────────────────


def _text(node: Any, path: str) -> str:
    found = node.find(path, _ATOM)
    return (found.text or "").strip() if found is not None and found.text else ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _year_from(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"(\d{4})", value)
    return int(m.group(1)) if m else None


def _ms_to_iso(ms: int | None) -> str:
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return ""


def _normalise_arxiv_id(value: str) -> str:
    m = _ARXIV_ID_RE.search(value)
    return m.group(1) if m else value.strip()


def _title_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
