"""Research synthesis: deduplicate, cluster, cross-link, assess maturity.

Runs with no network access at all (see the `research_synthesizer` contract). It can
only reason over what the source agents already collected and persisted, which is what
stops it from quietly becoming a ninth research agent producing evidence with no
provenance.

Maturity is computed from the *mix* of evidence, not from recency. A brand-new paper
with no implementation, no independent verification and no benchmark lands at
``speculative`` and routes to MONITOR — which is the behaviour the brief asks for when
it says a single new paper must not by itself mean the book changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from ...config import get_config
from ...research.models import (
    Claim,
    Contradiction,
    Evidence,
    EvidenceKind,
    EvidenceStrength,
    Maturity,
    ResearchCluster,
    SourceRef,
)
from ...tools import AgentContext
from ..base import Skill, array_of, boolean, json_schema, number, obj, string

CLUSTER_SCHEMA = json_schema(
    {
        "title": string("A precise technical title for this research theme"),
        "summary": string("4-6 sentences a textbook author could act on"),
        "concepts": array_of(string(), "Standard English concept names"),
        "key_claims": array_of(obj({
            "text": string(),
            "claim_type": string("", ["numerical", "benchmark", "historical", "causal",
                                      "sota", "definitional", "architectural",
                                      "attribution"]),
            "evidence_quality": string("", ["strong", "moderate", "weak", "anecdotal"]),
        }, ["text", "claim_type", "evidence_quality"])),
        "contradictions": array_of(obj({
            "statement_a": string(),
            "statement_b": string(),
            "nature": string("Why these conflict"),
        }, ["statement_a", "statement_b"])),
        "limitations": array_of(string()),
        "maturity": string(
            "speculative: one paper, no verification. emerging: replicated or "
            "implemented. consolidating: multiple independent sources plus real "
            "adoption. mature: settled, widely adopted, benchmarked.",
            ["speculative", "emerging", "consolidating", "mature", "superseded"]),
        "maturity_rationale": string("Why that maturity, citing the evidence mix"),
        "supersedes": array_of(string(), "Techniques this makes obsolete, if any"),
        "textbook_relevance": string("What a textbook would need to say, if anything"),
    },
    ["title", "summary", "concepts", "key_claims", "maturity", "maturity_rationale"],
)


class ResearchSynthesisSkill(Skill):
    name = "research_synthesis"
    required_tools = ("kb_query", "gemini_structured_output")
    optional_tools = ("gemini_embedding", "kb_upsert")

    async def run(
        self, ctx: AgentContext, *, items: list[dict[str, Any]],
        max_clusters: int | None = None, **_: Any,
    ) -> list[ResearchCluster]:
        cfg = get_config()
        if not items:
            return []

        deduped = await self._deduplicate(ctx, items)
        self.log.info(f"synthesis: {len(items)} items -> {len(deduped)} after dedup")

        groups = await self._cluster(ctx, deduped)
        max_clusters = max_clusters or int(cfg.get("synthesis.max_clusters_per_cycle", 12))
        groups = sorted(groups, key=lambda g: -_group_weight(g))[:max_clusters]
        self.log.info(f"synthesis: {len(groups)} clusters")

        clusters: list[ResearchCluster] = []
        for group in groups:
            cluster = await self._synthesise_group(ctx, group)
            if cluster:
                clusters.append(cluster)
        return clusters

    # -- deduplication -----------------------------------------------------
    async def _deduplicate(
        self, ctx: AgentContext, items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Three passes, cheapest first: identity, then DOI/arXiv, then embedding.

        Identity catches the same URL seen twice. DOI/arXiv catches the same paper
        arriving via arXiv and OpenAlex and HF daily papers — which happens constantly
        and would otherwise inflate a single result into three "independent" sources.
        """
        cfg = get_config()
        by_key: dict[str, dict[str, Any]] = {}

        for item in items:
            payload = item.get("payload") or item
            source = (payload.get("source") or {}).get("source") if isinstance(
                payload.get("source"), dict) else item.get("source", "")
            source_id = (payload.get("source") or {}).get("source_id") if isinstance(
                payload.get("source"), dict) else item.get("source_id", "")
            key = f"{source}|{source_id}"
            if key not in by_key:
                by_key[key] = item

        unique = list(by_key.values())

        # Identity across sources: the same paper reached us by several routes.
        by_identity: dict[str, dict[str, Any]] = {}
        standalone: list[dict[str, Any]] = []
        for item in unique:
            ident = _paper_identity(item)
            if ident:
                existing = by_identity.get(ident)
                if existing:
                    _merge_duplicate(existing, item)
                else:
                    by_identity[ident] = item
            else:
                standalone.append(item)
        unique = list(by_identity.values()) + standalone

        # Near-duplicate titles (preprint vs camera-ready, renamed versions).
        threshold = float(cfg.get("synthesis.dedupe_cosine_threshold", 0.86))
        if "gemini_embedding" in ctx.allowed_tools and len(unique) > 1:
            texts = [_item_text(i) for i in unique]
            emb = await ctx.try_call("gemini_embedding", default=None, texts=texts)
            if emb and emb.get("vectors"):
                vectors = emb["vectors"]
                keep: list[int] = []
                for i in range(len(unique)):
                    duplicate_of = None
                    for j in keep:
                        if _cosine(vectors[i], vectors[j]) >= threshold:
                            duplicate_of = j
                            break
                    if duplicate_of is None:
                        keep.append(i)
                    else:
                        _merge_duplicate(unique[duplicate_of], unique[i])
                for idx, item in enumerate(unique):
                    if idx in keep:
                        item["_vector"] = vectors[idx]
                unique = [unique[i] for i in keep]

        return unique

    # -- clustering --------------------------------------------------------
    async def _cluster(
        self, ctx: AgentContext, items: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group by concept overlap, refined by embedding similarity.

        Concept overlap alone over-splits (one paper says "GQA", another says
        "grouped-query attention"); embeddings alone over-merge everything about
        attention into one blob. Using concepts to seed and embeddings to join gets
        clusters a verdict agent can actually act on.
        """
        cfg = get_config()
        if not items:
            return []
        if len(items) == 1:
            return [items]

        threshold = float(cfg.get("synthesis.cluster_cosine_threshold", 0.74))
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for i, item in enumerate(items):
            if item.get("_vector"):
                vectors[i] = item["_vector"]
            else:
                missing.append(i)
        if missing and "gemini_embedding" in ctx.allowed_tools:
            emb = await ctx.try_call(
                "gemini_embedding", default=None,
                texts=[_item_text(items[i]) for i in missing])
            if emb and emb.get("vectors"):
                for idx, vec in zip(missing, emb["vectors"]):
                    vectors[idx] = vec

        concepts = [{c.lower().strip() for c in _item_concepts(it)} for it in items]

        def affinity(i: int, j: int) -> float:
            ci, cj = concepts[i], concepts[j]
            overlap = len(ci & cj) / max(1, min(len(ci), len(cj))) if ci and cj else 0.0
            sim = _cosine(vectors[i], vectors[j]) if i in vectors and j in vectors else 0.0
            return max(overlap, sim)

        # Assignment is against the whole cluster, not a single member. Union-find on
        # pairwise similarity is single-linkage, and single-linkage chains: A~B and
        # B~C merges A with C even when A and C are unrelated. A real cycle showed
        # exactly that — 23 items collapsed into one cluster titled "Structured Memory
        # Taxonomies and Speculative Decoding", two genuinely separate topics fused,
        # which then produces a verdict that is confused about what it is deciding.
        order = sorted(range(len(items)), key=lambda i: -len(concepts[i]))
        clusters: list[list[int]] = []

        for i in order:
            best_cluster, best_score = None, 0.0
            for cluster in clusters:
                scores = [affinity(i, j) for j in cluster]
                # Mean rather than max: the item must fit the cluster as a whole.
                score = sum(scores) / len(scores)
                if score > best_score:
                    best_cluster, best_score = cluster, score
            if best_cluster is not None and best_score >= threshold:
                best_cluster.append(i)
            else:
                clusters.append([i])

        min_size = int(cfg.get("synthesis.min_items_per_cluster", 1))
        groups = [[items[i] for i in cluster] for cluster in clusters]
        return [g for g in groups if len(g) >= min_size]

    # -- synthesis ---------------------------------------------------------
    async def _synthesise_group(
        self, ctx: AgentContext, group: list[dict[str, Any]],
    ) -> ResearchCluster | None:
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_evidence: list[Evidence] = []
        refs: list[SourceRef] = []

        for item in group:
            payload = item.get("payload") or item
            source = _item_source(item)
            by_role[_role_for(source, item)].append(item)
            for e in payload.get("evidence", []) or []:
                try:
                    all_evidence.append(Evidence.model_validate(e))
                except Exception:
                    continue
            try:
                refs.append(SourceRef.model_validate(payload["source"]))
            except Exception:
                pass

        digest = _evidence_digest(all_evidence)
        blob = "\n\n===\n".join(_item_detail(i) for i in group[:16])

        prompt = (
            "Synthesise these related research findings into one coherent theme for a "
            "graduate NLP/LLM textbook.\n\n"
            "CRITICAL — weigh evidence by kind, not by volume:\n"
            "  scientific              a peer-reviewed or preprint result\n"
            "  independent_verification someone other than the authors reproduced it\n"
            "  practical               it works in a real implementation\n"
            "  adoption                the field is actually using it\n"
            "  author_claim            the authors assert it; unverified\n"
            "  community               practitioners are talking about it\n"
            "  anecdotal_practical     one person's report\n\n"
            "Ten forum threads do not outweigh one replicated result, and one "
            "unreplicated paper is not a settled finding.\n\n"
            "Assign maturity from the evidence MIX, not from how recent the work is:\n"
            "  speculative    a single source, or author claims only, nothing verified\n"
            "  emerging       replicated OR implemented, but not both\n"
            "  consolidating  multiple independent sources AND real adoption\n"
            "  mature         settled, widely adopted, independently benchmarked\n\n"
            f"EVIDENCE MIX FOR THIS GROUP:\n{json.dumps(digest, indent=1)}\n\n"
            f"SOURCES ({len(group)} items):\n{blob[:44000]}"
        )

        result = await ctx.try_call(
            "gemini_structured_output", default=None,
            prompt=prompt, schema=CLUSTER_SCHEMA, temperature=0.2,
        )
        if not result:
            return None
        data = result["data"]

        cluster_id = "cl_" + hashlib.sha1(
            "|".join(sorted(_item_id(i) for i in group)).encode()).hexdigest()[:12]

        claims = [
            Claim(
                text=c.get("text", ""),
                claim_type=_safe_claim_type(c.get("claim_type")),
                confidence={"strong": 0.85, "moderate": 0.6,
                            "weak": 0.35, "anecdotal": 0.2}.get(
                    c.get("evidence_quality", "moderate"), 0.5),
                supporting_sources=refs[:5],
            )
            for c in (data.get("key_claims") or [])
        ]
        contradictions = [
            Contradiction(statement_a=c.get("statement_a", ""),
                          statement_b=c.get("statement_b", ""),
                          nature=c.get("nature", ""))
            for c in (data.get("contradictions") or [])
        ]

        stated = _safe_maturity(data.get("maturity"))
        computed = compute_maturity(all_evidence, group)
        # The model's maturity judgement is capped by what the evidence structurally
        # supports. Without this, an enthusiastic summary can promote a single
        # unreplicated preprint to "consolidating" and unlock a section rewrite.
        maturity = _min_maturity(stated, computed)

        return ResearchCluster(
            id=cluster_id,
            title=data.get("title", "Untitled research theme"),
            summary=data.get("summary", ""),
            concepts=(data.get("concepts") or [])[:12],
            papers=[_item_id(i) for i in by_role["paper"]],
            implementations=[_item_id(i) for i in by_role["implementation"]],
            benchmarks=[_item_id(i) for i in by_role["benchmark"]],
            community_signals=[_item_id(i) for i in by_role["community"]],
            blogs=[_item_id(i) for i in by_role["blog"]],
            claims=claims,
            evidence=all_evidence,
            contradictions=contradictions,
            limitations=(data.get("limitations") or [])[:10],
            maturity=maturity,
            provenance=refs,
        )


# -- maturity --------------------------------------------------------------

_MATURITY_ORDER = [Maturity.SPECULATIVE, Maturity.EMERGING,
                   Maturity.CONSOLIDATING, Maturity.MATURE]


def compute_maturity(evidence: list[Evidence], group: list[dict[str, Any]]) -> Maturity:
    """Ceiling on maturity, derived structurally from the evidence mix.

    Deliberately conservative and independent of the LLM: this is the guardrail that
    keeps "one exciting paper" from unlocking a section rewrite.
    """
    kinds = {e.kind for e in evidence}
    sources = {r.source for r in
               (p for e in evidence for p in e.provenance)}
    has_science = EvidenceKind.SCIENTIFIC in kinds
    has_independent = EvidenceKind.INDEPENDENT_VERIFICATION in kinds
    has_practical = EvidenceKind.PRACTICAL in kinds or EvidenceKind.ADOPTION in kinds
    strong_adoption = any(
        e.kind == EvidenceKind.ADOPTION and e.strength == EvidenceStrength.STRONG
        for e in evidence
    )
    distinct_sources = len(sources)

    if has_science and has_independent and strong_adoption and distinct_sources >= 4:
        return Maturity.MATURE
    if has_science and (has_independent or has_practical) and distinct_sources >= 3:
        return Maturity.CONSOLIDATING
    if has_science and (has_practical or len(group) >= 3):
        return Maturity.EMERGING
    if has_science or has_practical:
        return Maturity.EMERGING if distinct_sources >= 2 else Maturity.SPECULATIVE
    return Maturity.SPECULATIVE


def _min_maturity(a: Maturity, b: Maturity) -> Maturity:
    if a == Maturity.SUPERSEDED or b == Maturity.SUPERSEDED:
        return Maturity.SUPERSEDED
    return _MATURITY_ORDER[min(_MATURITY_ORDER.index(a), _MATURITY_ORDER.index(b))]


def _evidence_digest(evidence: list[Evidence]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_strength: dict[str, int] = {}
    for e in evidence:
        by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
        by_strength[e.strength.value] = by_strength.get(e.strength.value, 0) + 1
    sources = {p.source for e in evidence for p in e.provenance}
    return {"by_kind": by_kind, "by_strength": by_strength,
            "distinct_sources": sorted(sources), "total": len(evidence)}


# -- helpers ---------------------------------------------------------------


def _group_weight(group: list[dict[str, Any]]) -> float:
    weight = 0.0
    for item in group:
        payload = item.get("payload") or item
        weight += float(payload.get("relevance", item.get("relevance", 0.5)) or 0.5)
        weight += 0.25 * len(payload.get("evidence", []) or [])
    return weight + 0.5 * len(group)


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or (item.get("payload") or {}).get("id") or "")


def _item_source(item: dict[str, Any]) -> str:
    payload = item.get("payload") or item
    src = payload.get("source")
    if isinstance(src, dict):
        return str(src.get("source", ""))
    return str(item.get("source", src or ""))


#: Which slot of a cluster each source fills. The split matters because the verdict
#: agent reads "3 papers, 2 implementations, 1 benchmark" as a maturity signal —
#: three papers and nothing else is a very different situation from three papers
#: plus a maintained implementation.
_SOURCE_ROLES = {
    "arxiv": "paper",
    "openalex": "paper",
    "semantic_scholar": "paper",
    "crossref": "paper",
    "datacite": "paper",
    "openreview": "paper",
    "acl": "paper",
    "github": "implementation",
    "huggingface": "implementation",
    "hf_papers": "paper",
    "blog": "blog",
    "hackernews": "community",
    "github_discussion": "community",
    "reddit": "community",
    "web": "blog",
}


def _role_for(source: str, item: dict[str, Any] | None = None) -> str:
    # A benchmark repository arrives from GitHub like any other repo, but it is
    # evidence of a different kind — the Benchmark Agent tags it so the cluster can
    # tell "someone implemented this" from "someone measured this".
    if item is not None:
        payload = item.get("payload") or item
        raw = payload.get("raw") or {}
        if raw.get("benchmark_name") or item.get("benchmark_name"):
            return "benchmark"
    return _SOURCE_ROLES.get(source, "blog")


def _item_concepts(item: dict[str, Any]) -> list[str]:
    payload = item.get("payload") or item
    concepts = payload.get("concepts") or item.get("concepts") or []
    if isinstance(concepts, str):
        try:
            concepts = json.loads(concepts)
        except json.JSONDecodeError:
            concepts = [concepts]
    return [str(c) for c in concepts]


def _item_text(item: dict[str, Any]) -> str:
    payload = item.get("payload") or item
    return (f"{payload.get('title', item.get('title', ''))}. "
            f"{payload.get('summary', item.get('summary', ''))} "
            f"{' '.join(_item_concepts(item))}")[:3000]


def _item_detail(item: dict[str, Any]) -> str:
    payload = item.get("payload") or item
    src = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    lines = [
        f"SOURCE: {src.get('source', _item_source(item))}",
        f"TITLE: {payload.get('title', item.get('title', ''))}",
        f"URL: {src.get('url', '')}",
        f"DATE: {payload.get('published_at', item.get('published_at', ''))}",
        f"SUMMARY: {payload.get('summary', item.get('summary', ''))}",
        f"CONCEPTS: {', '.join(_item_concepts(item)[:10])}",
    ]
    if payload.get("methods"):
        lines.append(f"METHODS: {', '.join(str(m) for m in payload['methods'][:8])}")
    for b in (payload.get("benchmarks") or [])[:6]:
        lines.append(f"BENCHMARK: {b.get('benchmark')} {b.get('model','')} "
                     f"{b.get('value','')} {b.get('metric','')} "
                     f"(vs {b.get('comparison','?')})")
    for e in (payload.get("evidence") or [])[:8]:
        lines.append(f"EVIDENCE[{e.get('kind')}/{e.get('strength')}]: "
                     f"{str(e.get('statement',''))[:280]}")
    if payload.get("limitations"):
        lines.append(f"LIMITATIONS: {'; '.join(str(x) for x in payload['limitations'][:5])}")
    raw = payload.get("raw") or {}
    for field in ("stars", "downloads", "maintenance", "adoption_signal",
                  "interest_level", "engagement", "citation_count"):
        if raw.get(field) is not None:
            lines.append(f"{field.upper()}: {raw[field]}")
    return "\n".join(lines)


def _paper_identity(item: dict[str, Any]) -> str:
    """DOI or arXiv id, normalised — the only reliable cross-source paper identity."""
    payload = item.get("payload") or item
    raw = payload.get("raw") or {}
    for field in ("doi", "arxiv_id"):
        value = str(raw.get(field) or payload.get(field) or item.get(field) or "").strip()
        if value:
            value = value.lower().replace("https://doi.org/", "")
            value = value.replace("10.48550/arxiv.", "")
            return f"id:{value}"
    return ""


def _merge_duplicate(keeper: dict[str, Any], other: dict[str, Any]) -> None:
    """Fold a duplicate's evidence into the keeper, preserving both provenances.

    The duplicate is not discarded outright: the fact that a result reached us via
    two independent routes is itself signal, and its evidence entries carry distinct
    provenance the verdict agent should see.
    """
    kp = keeper.setdefault("payload", {k: v for k, v in keeper.items() if k != "payload"})
    op = other.get("payload") or other
    kp.setdefault("evidence", [])
    existing = {json.dumps(e, sort_keys=True, default=str) for e in kp["evidence"]}
    for e in op.get("evidence", []) or []:
        blob = json.dumps(e, sort_keys=True, default=str)
        if blob not in existing:
            kp["evidence"].append(e)
            existing.add(blob)
    kp.setdefault("duplicate_of", []).append(_item_id(other))
    concepts = set(kp.get("concepts") or []) | set(op.get("concepts") or [])
    kp["concepts"] = sorted(concepts)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _safe_claim_type(value: Any):
    from ...research.models import ClaimType
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.DEFINITIONAL


def _safe_maturity(value: Any) -> Maturity:
    try:
        return Maturity(str(value))
    except ValueError:
        return Maturity.SPECULATIVE
