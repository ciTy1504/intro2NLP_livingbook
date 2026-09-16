"""Research discovery and per-source analysis skills.

The pipeline each source agent follows is the same shape —

    discover -> retrieve -> extract -> normalise -> attach provenance

— but the *semantics* differ per source, which is why these are skills parameterised
by source rather than one generic research agent. A paper's claims are its authors'
assertions; a repository's star count is an adoption fact; a forum thread is a report
of somebody's experience. Collapsing those into "results" is what produces a book that
cites Reddit for a benchmark number.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ...config import get_config
from ...research.models import (
    SOURCE_EVIDENCE_KINDS,
    BenchmarkResult,
    Claim,
    ClaimType,
    Evidence,
    EvidenceKind,
    EvidenceStrength,
    ResearchItem,
    SourceRef,
)
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, integer, json_schema, number, obj, string

RELEVANCE_SCHEMA = json_schema(
    {
        "relevant": boolean("Would this plausibly affect an NLP/LLM textbook?"),
        "relevance": number("0..1"),
        "reason": string("One sentence"),
        "concepts": array_of(string(), "Standard English technical concepts, max 6"),
        "book_topics": array_of(string(), "Which book topics this touches"),
    },
    ["relevant", "relevance", "reason", "concepts"],
)

PAPER_ANALYSIS_SCHEMA = json_schema(
    {
        "summary": string("3-5 sentences: what the paper does and why it matters"),
        "concepts": array_of(string(), "Standard English concept names"),
        "methods": array_of(string(), "Techniques introduced or used"),
        "claims": array_of(obj({
            "text": string("A checkable claim the paper makes, self-contained"),
            "claim_type": string("", ["numerical", "benchmark", "historical", "causal",
                                      "sota", "definitional", "architectural",
                                      "attribution"]),
            "is_author_claim": boolean("True unless independently verified in the paper"),
        }, ["text", "claim_type", "is_author_claim"])),
        "benchmarks": array_of(obj({
            "benchmark": string(),
            "metric": string(),
            "value": string(),
            "model": string(),
            "comparison": string("What it is compared against"),
            "is_sota_claim": boolean(),
        }, ["benchmark"])),
        "limitations": array_of(string(), "Limitations the paper itself states"),
        "related_work": array_of(string(), "Key prior work it builds on"),
        "novelty": string("What is genuinely new here"),
        "maturity_signal": string("", ["speculative", "emerging", "consolidating", "mature"]),
    },
    ["summary", "concepts", "methods", "claims", "limitations"],
)

REPO_ANALYSIS_SCHEMA = json_schema(
    {
        "summary": string("What this implements and who would use it"),
        "concepts": array_of(string()),
        "implements": array_of(string(), "Techniques/papers this implements"),
        "adoption_signal": string("", ["strong", "moderate", "weak"]),
        "maintenance": string("", ["active", "maintained", "stale", "abandoned"]),
        "practical_notes": array_of(string(), "Practical facts a textbook could use"),
        "is_reference_implementation": boolean(),
    },
    ["summary", "concepts", "adoption_signal", "maintenance"],
)

COMMUNITY_ANALYSIS_SCHEMA = json_schema(
    {
        "summary": string("What practitioners are actually saying"),
        "concepts": array_of(string()),
        "recurring_themes": array_of(string()),
        "emerging_terminology": array_of(string()),
        "pain_points": array_of(string()),
        "disagreements": array_of(string()),
        "interest_level": string("", ["high", "moderate", "low"]),
        "contains_verifiable_claims": boolean(
            "True only if a specific checkable claim is made, not just opinion"),
    },
    ["summary", "concepts", "recurring_themes", "interest_level"],
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ResearchDiscoverySkill(Skill):
    """Query a source, filter to what could plausibly matter to this book."""

    name = "research_discovery"
    optional_tools = (
        "search_arxiv", "search_openalex", "search_github", "search_huggingface",
        "search_openreview", "search_acl_anthology", "search_hackernews",
        "search_hf_papers", "search_github_discussions", "search_reddit",
        "fetch_rss", "web_search", "search_semantic_scholar", "search_crossref",
    )

    async def run(
        self, ctx: AgentContext, *, tool: str, tool_args: dict[str, Any],
        max_items: int = 40, **_: Any,
    ) -> list[dict[str, Any]]:
        if tool not in ctx.allowed_tools:
            self.log.debug(f"{ctx.agent}: {tool} not permitted, skipping")
            return []
        # try_call so an unavailable source degrades the cycle instead of failing it.
        raw = await ctx.try_call(tool, default=[], **tool_args)
        items = raw if isinstance(raw, list) else [raw] if raw else []
        return items[:max_items]

    async def screen(
        self, ctx: AgentContext, items: list[dict[str, Any]], *,
        min_relevance: float = 0.35,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Cheap relevance screen before paying for deep analysis.

        Deep analysis costs a large model call and possibly a PDF download per item.
        Screening first on title + abstract with the `fast` role keeps a 40-item
        discovery batch affordable.
        """
        cfg = get_config()
        topics = cfg.get("research.topics", [])
        kept: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for item in items:
            blurb = _blurb(item)
            if not blurb.strip():
                continue
            prompt = (
                "You are screening research for a Vietnamese textbook on NLP and large "
                "language models.\n\n"
                f"The book covers: {', '.join(topics)}.\n\n"
                "Decide whether this item could plausibly affect the book's content. "
                "Be generous at this stage — a later agent decides whether the book "
                "actually changes. Reject only clearly unrelated work.\n\n"
                f"ITEM:\n{blurb[:6000]}"
            )
            result = await ctx.try_call(
                "gemini_structured_output", default=None,
                prompt=prompt, schema=RELEVANCE_SCHEMA, role="fast", temperature=0.1,
            )
            if not result:
                continue
            data = result["data"]
            if data.get("relevant") and float(data.get("relevance", 0)) >= min_relevance:
                kept.append((item, data))
        return kept


class PaperAnalysisSkill(Skill):
    """Extract methods, claims, benchmarks and limitations from a paper."""

    name = "paper_analysis"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("download_pdf", "parse_pdf", "fetch_arxiv_paper",
                      "fetch_openalex_work", "fetch_url")

    async def run(
        self, ctx: AgentContext, *, item: dict[str, Any], screen: dict[str, Any] | None = None,
        deep: bool = False, **_: Any,
    ) -> ResearchItem | None:
        source_name = item.get("source", "web")
        text = _blurb(item)

        # A full-text read is expensive; reserve it for items that cleared screening
        # with high relevance, where the extra detail actually changes the outcome.
        if deep and item.get("pdf_url") and "parse_pdf" in ctx.allowed_tools:
            parsed = await ctx.try_call("parse_pdf", default=None, url=item["pdf_url"])
            if parsed and parsed.get("text"):
                sections = parsed.get("sections", {})
                interesting = " ".join(
                    sections.get(k, "")[:6000]
                    for k in ("abstract", "introduction", "results", "evaluation",
                              "limitations", "conclusion")
                    if sections.get(k)
                )
                text = (text + "\n\n" + (interesting or parsed["text"][:24000]))[:40000]

        prompt = (
            "Analyse this research for inclusion in a graduate-level NLP/LLM textbook.\n\n"
            "Be precise about what is CLAIMED versus what is SHOWN. Mark a claim as "
            "an author claim unless the paper reports independent verification.\n"
            "Report benchmark numbers exactly as stated, with what they are compared "
            "against — a number without its comparison is not usable in a textbook.\n"
            "Answer in English.\n\n"
            f"SOURCE: {source_name}\n{text[:40000]}"
        )
        result = await ctx.try_call(
            "gemini_structured_output", default=None,
            prompt=prompt, schema=PAPER_ANALYSIS_SCHEMA, temperature=0.15,
        )
        if not result:
            return None
        data = result["data"]

        ref = _source_ref(item)
        allowed = _allowed_kinds(ctx, source_name)

        claims: list[Claim] = []
        evidence: list[Evidence] = []
        for c in data.get("claims", []) or []:
            claims.append(Claim(
                text=c.get("text", ""),
                claim_type=_claim_type(c.get("claim_type")),
                needs_citation=True,
                supporting_sources=[ref],
            ))
            # An author's own claim is not a verified result, and the distinction is
            # what stops the book asserting a number the field has not confirmed.
            kind = (EvidenceKind.AUTHOR_CLAIM if c.get("is_author_claim", True)
                    else EvidenceKind.SCIENTIFIC)
            evidence.append(Evidence.for_source(
                source_name, kind, c.get("text", ""),
                strength=(EvidenceStrength.MODERATE if c.get("is_author_claim", True)
                          else EvidenceStrength.STRONG),
                provenance=[ref], supports=c.get("text", ""), allowed_kinds=allowed,
            ))

        benchmarks = [
            BenchmarkResult(**{k: str(v) if not isinstance(v, bool) else v
                               for k, v in b.items() if k in BenchmarkResult.model_fields})
            for b in (data.get("benchmarks", []) or [])
            if b.get("benchmark")
        ]
        for b in benchmarks:
            evidence.append(Evidence.for_source(
                source_name, EvidenceKind.SCIENTIFIC,
                f"{b.benchmark}: {b.model or 'model'} = {b.value} {b.metric}".strip(),
                strength=EvidenceStrength.STRONG if not b.is_sota_claim
                else EvidenceStrength.MODERATE,
                provenance=[ref], allowed_kinds=allowed,
            ))

        return ResearchItem(
            id=ResearchItem.make_id(source_name, str(item.get("source_id", ""))),
            source=ref,
            title=item.get("title", ""),
            summary=data.get("summary", ""),
            published_at=item.get("published") or item.get("published_at"),
            concepts=(data.get("concepts") or [])[:12],
            methods=(data.get("methods") or [])[:12],
            benchmarks=benchmarks,
            claims=claims,
            limitations=(data.get("limitations") or [])[:10],
            related_work=(data.get("related_work") or [])[:10],
            evidence=evidence,
            relevance=float((screen or {}).get("relevance", 0.5)),
            raw={"novelty": data.get("novelty", ""),
                 "maturity_signal": data.get("maturity_signal", "emerging"),
                 **{k: v for k, v in item.items() if k in
                    ("url", "doi", "arxiv_id", "venue", "citation_count", "authors")}},
        )


class RepoAnalysisSkill(Skill):
    """Read a repository or model as an adoption and practicality signal."""

    name = "repo_analysis"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("fetch_github_repo", "fetch_github_activity", "fetch_hf_model",
                      "fetch_url")

    async def run(
        self, ctx: AgentContext, *, item: dict[str, Any], screen: dict[str, Any] | None = None,
        with_activity: bool = True, **_: Any,
    ) -> ResearchItem | None:
        source_name = item.get("source", "github")
        detail = dict(item)

        if source_name == "github" and "fetch_github_repo" in ctx.allowed_tools:
            full = await ctx.try_call("fetch_github_repo", default=None,
                                      full_name=item.get("source_id", ""))
            if full:
                detail.update(full)
            if with_activity and "fetch_github_activity" in ctx.allowed_tools:
                act = await ctx.try_call("fetch_github_activity", default=None,
                                         full_name=item.get("source_id", ""))
                if act:
                    detail["activity"] = act
        elif source_name == "huggingface" and "fetch_hf_model" in ctx.allowed_tools:
            full = await ctx.try_call("fetch_hf_model", default=None,
                                      model_id=item.get("source_id", ""))
            if full:
                detail.update(full)

        prompt = (
            "Assess this implementation as evidence for an NLP/LLM textbook.\n\n"
            "This is an ADOPTION and PRACTICALITY signal, not a scientific result. "
            "Popularity shows a technique is used, not that it works better than "
            "alternatives. Judge maintenance from recent commits and releases, not "
            "from star count. Answer in English.\n\n"
            f"{_repo_blurb(detail)[:24000]}"
        )
        result = await ctx.try_call(
            "gemini_structured_output", default=None,
            prompt=prompt, schema=REPO_ANALYSIS_SCHEMA, temperature=0.15,
        )
        if not result:
            return None
        data = result["data"]

        ref = _source_ref(detail)
        allowed = _allowed_kinds(ctx, source_name)
        strength = {"strong": EvidenceStrength.STRONG,
                    "moderate": EvidenceStrength.MODERATE,
                    "weak": EvidenceStrength.WEAK}.get(
            data.get("adoption_signal", "moderate"), EvidenceStrength.MODERATE)

        evidence = [Evidence.for_source(
            source_name, EvidenceKind.ADOPTION,
            f"{detail.get('title', '')}: {data.get('adoption_signal')} adoption, "
            f"{data.get('maintenance')} maintenance"
            + (f", {detail.get('stars')} stars" if detail.get("stars") else "")
            + (f", {detail.get('downloads')} downloads" if detail.get("downloads") else ""),
            strength=strength, provenance=[ref], allowed_kinds=allowed,
        )]
        for note in (data.get("practical_notes") or [])[:6]:
            evidence.append(Evidence.for_source(
                source_name, EvidenceKind.PRACTICAL, note,
                strength=EvidenceStrength.MODERATE, provenance=[ref],
                allowed_kinds=allowed,
            ))

        return ResearchItem(
            id=ResearchItem.make_id(source_name, str(detail.get("source_id", ""))),
            source=ref,
            title=detail.get("title", ""),
            summary=data.get("summary", ""),
            published_at=detail.get("created_at") or detail.get("pushed_at"),
            concepts=(data.get("concepts") or [])[:12],
            methods=(data.get("implements") or [])[:10],
            evidence=evidence,
            relevance=float((screen or {}).get("relevance", 0.5)),
            raw={
                "stars": detail.get("stars"), "downloads": detail.get("downloads"),
                "license": detail.get("license"), "url": detail.get("url"),
                "maintenance": data.get("maintenance"),
                "adoption_signal": data.get("adoption_signal"),
                "is_reference_implementation": data.get("is_reference_implementation"),
                "arxiv_ids": detail.get("arxiv_ids", []),
                "activity": detail.get("activity", {}),
            },
        )


class CommunitySignalSkill(Skill):
    """Read practitioner discussion. Produces signal; never scientific evidence."""

    name = "community_signal_analysis"
    required_tools = ("gemini_structured_output",)
    optional_tools = ("search_hackernews", "fetch_hn_thread", "search_github_discussions")

    async def run(
        self, ctx: AgentContext, *, items: list[dict[str, Any]], topic: str = "", **_: Any,
    ) -> ResearchItem | None:
        if not items:
            return None
        source_name = items[0].get("source", "hackernews")
        blob = "\n\n---\n".join(
            f"[{it.get('points', it.get('reactions', 0))} pts, "
            f"{it.get('num_comments', it.get('comments', 0))} comments] "
            f"{it.get('title','')}\n{(it.get('text') or it.get('body') or '')[:1500]}"
            for it in items[:12]
        )

        prompt = (
            "Summarise what practitioners are discussing, for a textbook author "
            "deciding whether a topic deserves coverage.\n\n"
            "This is COMMUNITY SIGNAL. It tells you what people find interesting, "
            "confusing or painful — not what is true. Do not treat any assertion here "
            "as an established result. Answer in English.\n\n"
            f"TOPIC: {topic}\n\nDISCUSSIONS:\n{blob[:24000]}"
        )
        result = await ctx.try_call(
            "gemini_structured_output", default=None,
            prompt=prompt, schema=COMMUNITY_ANALYSIS_SCHEMA, temperature=0.2,
        )
        if not result:
            return None
        data = result["data"]

        refs = [_source_ref(it) for it in items[:12]]
        allowed = _allowed_kinds(ctx, source_name)
        total_engagement = sum(
            int(it.get("points", 0) or 0) + int(it.get("num_comments", 0) or 0)
            + int(it.get("reactions", 0) or 0) for it in items
        )
        strength = (EvidenceStrength.MODERATE
                    if data.get("interest_level") == "high" and len(items) >= 3
                    else EvidenceStrength.ANECDOTAL)

        evidence = [Evidence.for_source(
            source_name, EvidenceKind.COMMUNITY,
            f"{len(items)} discussions ({total_engagement} total engagement) on "
            f"{topic or 'this topic'}: {data.get('summary','')[:400]}",
            strength=strength, provenance=refs, allowed_kinds=allowed,
        )]
        for pain in (data.get("pain_points") or [])[:5]:
            evidence.append(Evidence.for_source(
                source_name, EvidenceKind.ANECDOTAL_PRACTICAL, pain,
                strength=EvidenceStrength.ANECDOTAL, provenance=refs[:3],
                allowed_kinds=allowed,
            ))

        digest = hashlib.sha1(
            f"{topic}|{'|'.join(r.source_id for r in refs)}".encode()).hexdigest()[:12]
        return ResearchItem(
            id=f"ri_{source_name}_{digest}",
            source=SourceRef(
                source=source_name, source_id=f"digest:{digest}",
                url=items[0].get("discussion_url") or items[0].get("url", ""),
                title=f"Community discussion: {topic}" if topic else "Community discussion",
                content_sha256=_sha(blob),
            ),
            title=f"Community signal: {topic or data.get('summary','')[:60]}",
            summary=data.get("summary", ""),
            published_at=items[0].get("published"),
            concepts=(data.get("concepts") or [])[:10],
            evidence=evidence,
            relevance=0.45 if data.get("interest_level") == "high" else 0.3,
            raw={
                "recurring_themes": data.get("recurring_themes", []),
                "emerging_terminology": data.get("emerging_terminology", []),
                "pain_points": data.get("pain_points", []),
                "disagreements": data.get("disagreements", []),
                "interest_level": data.get("interest_level"),
                "engagement": total_engagement,
                "thread_count": len(items),
            },
        )


# -- helpers ---------------------------------------------------------------


def _allowed_kinds(ctx: AgentContext, source: str) -> set[EvidenceKind]:
    """Intersect what the source can support with what the agent may emit."""
    by_source = SOURCE_EVIDENCE_KINDS.get(source, {EvidenceKind.COMMUNITY})
    declared = ctx.extra.get("evidence_kinds") or []
    if not declared:
        return by_source
    by_agent = set()
    for k in declared:
        try:
            by_agent.add(EvidenceKind(k))
        except ValueError:
            continue
    return (by_source & by_agent) or by_source


def _source_ref(item: dict[str, Any]) -> SourceRef:
    body = str(item.get("abstract") or item.get("summary") or item.get("description")
               or item.get("text") or "")
    return SourceRef(
        source=str(item.get("source", "web")),
        source_id=str(item.get("source_id", item.get("url", ""))),
        url=str(item.get("url", "")),
        title=str(item.get("title", "")),
        published_at=item.get("published") or item.get("published_at"),
        content_sha256=_sha(f"{item.get('title','')}|{body}") if body else None,
    )


def _blurb(item: dict[str, Any]) -> str:
    parts = [f"Title: {item.get('title', '')}"]
    if item.get("authors"):
        parts.append(f"Authors: {', '.join(str(a) for a in item['authors'][:8])}")
    if item.get("venue"):
        parts.append(f"Venue: {item['venue']}")
    if item.get("year") or item.get("published"):
        parts.append(f"Date: {item.get('year') or item.get('published')}")
    if item.get("citation_count") is not None:
        parts.append(f"Citations: {item['citation_count']}")
    body = (item.get("abstract") or item.get("summary") or item.get("description")
            or item.get("text") or "")
    if body:
        parts.append(f"\n{body}")
    return "\n".join(parts)


def _repo_blurb(item: dict[str, Any]) -> str:
    parts = [
        f"Name: {item.get('title', item.get('source_id', ''))}",
        f"Description: {item.get('description', '')}",
        f"Stars: {item.get('stars', 'n/a')}  Forks: {item.get('forks', 'n/a')}  "
        f"Downloads: {item.get('downloads', 'n/a')}  Likes: {item.get('likes', 'n/a')}",
        f"Language: {item.get('language', 'n/a')}  License: {item.get('license', 'n/a')}",
        f"Created: {item.get('created_at', 'n/a')}  Last push: {item.get('pushed_at', 'n/a')}",
        f"Topics/tags: {', '.join(str(t) for t in (item.get('topics') or item.get('tags') or [])[:12])}",
        f"Archived: {item.get('archived', False)}",
    ]
    activity = item.get("activity") or {}
    if activity:
        parts.append(
            f"Commits in last {activity.get('window_days', 90)}d: "
            f"{activity.get('commits_in_window')}  "
            f"Issues opened: {activity.get('issues_opened_in_window')}"
        )
        for rel in (activity.get("releases") or [])[:3]:
            parts.append(f"Release {rel.get('tag')} ({rel.get('published_at')}): "
                         f"{(rel.get('body') or '')[:300]}")
    if item.get("arxiv_ids"):
        parts.append(f"Linked papers: {', '.join(item['arxiv_ids'][:5])}")
    readme = item.get("readme") or item.get("model_card") or ""
    if readme:
        parts.append(f"\nREADME:\n{readme[:12000]}")
    return "\n".join(parts)


def _claim_type(value: Any) -> ClaimType:
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.DEFINITIONAL
