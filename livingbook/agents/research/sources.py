"""Source-specific research agents.

One agent per source, because sources differ in what they can tell you. arXiv reports
what authors claim; GitHub reports what people build; Hacker News reports what people
find painful. A single "ResearchAgent" would have to flatten those into one notion of
"a result", which is how a textbook ends up citing a forum post for a benchmark number.

Each agent runs the same shape — discover, retrieve, extract, normalise, attach
provenance — and writes into Research Memory, where the synthesiser picks it up.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from ...config import get_config
from ...research.models import ResearchItem
from ...state.store import utcnow
from ..base import BaseAgent


class SourceAgent(BaseAgent[list[ResearchItem]]):
    """Shared behaviour for the eight source agents."""

    uses_skills: tuple[str, ...] = ("research_discovery",)
    #: "paper" | "implementation" | "benchmark" | "community" | "blog"
    item_role: str = "paper"
    #: Only fetch full text for items that screened above this relevance.
    deep_threshold: float = 0.7

    def degraded_result(self) -> list[ResearchItem]:
        return []

    async def execute(self, *, since_days: int | None = None,
                      max_items: int | None = None, **_: Any) -> list[ResearchItem]:
        cfg = get_config()
        since_days = since_days or int(cfg.get("research.lookback_days", 21))
        max_items = max_items or int(cfg.get("research.max_items_per_source_per_cycle", 40))

        raw = await self.discover(since_days=since_days, max_items=max_items)
        if not raw:
            self.log.info(f"{self.name}: nothing discovered")
            return []

        fresh = self._drop_already_seen(raw)
        self.log.info(
            f"{self.name}: {len(raw)} found, {len(fresh)} new "
            f"({len(raw) - len(fresh)} already in research memory)")
        if not fresh:
            return []

        discovery = self.skill("research_discovery")
        screened = await discovery.screen(self.ctx, fresh[:max_items])
        self.log.info(f"{self.name}: {len(screened)} passed relevance screening")
        if not screened:
            return []

        items = await self.analyse(screened)
        items = [i for i in items if i]
        await self.persist(items)
        self.log.info(f"{self.name}: {len(items)} research items recorded", status="ok")
        return items

    # -- steps to override -------------------------------------------------
    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def analyse(
        self, screened: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[ResearchItem | None]:
        """Default: analyse each item as a paper, with bounded concurrency."""
        skill = self.skill("paper_analysis")
        sem = asyncio.Semaphore(4)

        async def one(item: dict[str, Any], screen: dict[str, Any]) -> ResearchItem | None:
            async with sem:
                return await skill(
                    self.ctx, item=item, screen=screen,
                    deep=float(screen.get("relevance", 0)) >= self.deep_threshold)

        results = await asyncio.gather(
            *(one(i, s) for i, s in screened), return_exceptions=True)
        out: list[ResearchItem | None] = []
        for r in results:
            if isinstance(r, Exception):
                self.log.warn(f"{self.name}: analysis failed: {type(r).__name__}: {r}")
            else:
                out.append(r)
        return out

    # -- shared plumbing ---------------------------------------------------
    def _drop_already_seen(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Skip anything already in Research Memory.

        Sources return the same items every cycle. Re-analysing them would burn the
        key pool for nothing and inflate a repeated result into apparent corroboration.
        """
        if not raw:
            return []
        known: set[str] = set()
        rows = self.store.query(
            "SELECT source, source_id FROM research_items WHERE source = ?",
            (self._source_name(),))
        for r in rows:
            known.add(str(r["source_id"]))

        fresh = []
        touched: list[str] = []
        for item in raw:
            sid = str(item.get("source_id", ""))
            if sid and sid in known:
                touched.append(sid)
                continue
            fresh.append(item)

        if touched:
            # Re-seeing an item is itself information: it keeps last_seen_at current
            # so the monitor sweep can distinguish a dormant topic from a live one.
            self.store.executemany(
                "UPDATE research_items SET last_seen_at = ? "
                "WHERE source = ? AND source_id = ?",
                [(utcnow(), self._source_name(), sid) for sid in touched])
        return fresh

    def _source_name(self) -> str:
        return getattr(self, "source_name", self.name.replace("_agent", ""))

    async def persist(self, items: list[ResearchItem]) -> None:
        if not items:
            return
        await self.ctx.call(
            "kb_upsert", kind="research_item",
            records=[i.to_row() for i in items])

        evidence_rows = []
        for item in items:
            for n, e in enumerate(item.evidence):
                evidence_rows.append({
                    "id": f"ev_{item.id}_{n}",
                    "research_item_id": item.id,
                    "kind": e.kind.value,
                    "statement": e.statement,
                    "strength": e.strength.value,
                    "supports": e.supports,
                    "provenance": [p.model_dump(mode="json") for p in e.provenance],
                })
        if evidence_rows:
            await self.ctx.call("kb_upsert", kind="evidence", records=evidence_rows)

    @staticmethod
    def _cutoff(days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


class ArxivAgent(SourceAgent):
    name = "arxiv_agent"
    source_name = "arxiv"
    item_role = "paper"
    uses_skills = ("research_discovery", "paper_analysis", "claim_extraction")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        cfg = get_config()
        topics = cfg.get("research.topics", [])
        categories = cfg.get("research.arxiv_categories", ["cs.CL"])
        discovery = self.skill("research_discovery")

        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Rotate topics by day so every topic is covered over time without issuing
        # twenty queries per cycle against a source that asks for one request per 3s.
        rotation = _rotate(topics, per_cycle=6)
        for topic in rotation:
            items = await discovery(
                self.ctx, tool="search_arxiv",
                tool_args={"query": topic, "categories": categories,
                           "max_results": 12, "since_days": since_days},
                max_items=12)
            for it in items:
                if it.get("source_id") not in seen:
                    seen.add(it.get("source_id"))
                    found.append(it)
            if len(found) >= max_items:
                break
        return found[:max_items]


class ScholarlyAgent(SourceAgent):
    name = "scholarly_agent"
    source_name = "openalex"
    item_role = "paper"
    uses_skills = ("research_discovery", "paper_analysis")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        cfg = get_config()
        discovery = self.skill("research_discovery")
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        for topic in _rotate(cfg.get("research.topics", []), per_cycle=5):
            items = await discovery(
                self.ctx, tool="search_openalex",
                tool_args={"query": topic, "per_page": 10, "since_days": since_days},
                max_items=10)
            for it in items:
                if it.get("source_id") not in seen:
                    seen.add(it.get("source_id"))
                    found.append(it)
            if len(found) >= max_items:
                break
        return found[:max_items]


class GithubResearchAgent(SourceAgent):
    name = "github_research_agent"
    source_name = "github"
    item_role = "implementation"
    uses_skills = ("research_discovery", "repo_analysis")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        cfg = get_config()
        min_stars = int(cfg.get("research.sources.github.min_stars", 150))
        discovery = self.skill("research_discovery")
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        for query in _rotate(cfg.source("github.search_queries") or [], per_cycle=4):
            items = await discovery(
                self.ctx, tool="search_github",
                tool_args={"query": query, "limit": 8, "min_stars": min_stars,
                           "pushed_days": max(since_days, 60)},
                max_items=8)
            for it in items:
                if it.get("source_id") not in seen and not it.get("archived"):
                    seen.add(it.get("source_id"))
                    found.append(it)

        # Watched repositories are checked every cycle: a release in vLLM or
        # transformers is a stronger adoption signal than a new repo with 200 stars.
        for repo in (cfg.source("github.watch_repos") or [])[:10]:
            if repo in seen:
                continue
            info = await self.ctx.try_call("fetch_github_repo", default=None,
                                           full_name=repo, with_readme=False)
            if info and _recently_active(info.get("pushed_at"), since_days):
                seen.add(repo)
                found.append(info)
        return found[:max_items]

    async def analyse(self, screened):  # type: ignore[override]
        skill = self.skill("repo_analysis")
        sem = asyncio.Semaphore(3)

        async def one(item, screen):
            async with sem:
                return await skill(self.ctx, item=item, screen=screen)

        results = await asyncio.gather(*(one(i, s) for i, s in screened),
                                       return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]


class HuggingFaceAgent(SourceAgent):
    name = "huggingface_agent"
    source_name = "huggingface"
    item_role = "implementation"
    uses_skills = ("research_discovery", "repo_analysis")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        discovery = self.skill("research_discovery")
        found: list[dict[str, Any]] = []

        papers = await discovery(
            self.ctx, tool="search_hf_papers",
            tool_args={"limit": 30, "days": since_days}, max_items=20)
        found.extend(papers or [])

        for topic in _rotate(get_config().get("research.topics", []), per_cycle=3):
            models = await discovery(
                self.ctx, tool="search_huggingface",
                tool_args={"query": topic, "kind": "models", "limit": 6},
                max_items=6)
            found.extend(models or [])
        return found[:max_items]

    async def analyse(self, screened):  # type: ignore[override]
        # Daily-papers entries are papers; models are implementations. The same agent
        # legitimately produces both, and they need different analysis.
        papers = [(i, s) for i, s in screened if i.get("source") == "hf_papers"]
        models = [(i, s) for i, s in screened if i.get("source") != "hf_papers"]
        out: list[ResearchItem | None] = []

        if papers:
            skill = self.skill("repo_analysis")  # HF agent has no paper_analysis grant
            for item, screen in papers:
                item = {**item, "source": "huggingface", "description": item.get("abstract", "")}
                out.append(await skill(self.ctx, item=item, screen=screen,
                                       with_activity=False))
        if models:
            skill = self.skill("repo_analysis")
            sem = asyncio.Semaphore(3)

            async def one(item, screen):
                async with sem:
                    return await skill(self.ctx, item=item, screen=screen,
                                       with_activity=False)

            results = await asyncio.gather(*(one(i, s) for i, s in models),
                                           return_exceptions=True)
            out.extend(r for r in results if not isinstance(r, Exception))
        return out


class ResearchBlogAgent(SourceAgent):
    name = "research_blog_agent"
    source_name = "blog"
    item_role = "blog"
    uses_skills = ("research_discovery", "paper_analysis")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        feeds = get_config().source("blogs.feeds") or []
        discovery = self.skill("research_discovery")
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        found: list[dict[str, Any]] = []

        for feed in feeds:
            entries = await discovery(
                self.ctx, tool="fetch_rss",
                tool_args={"url": feed["url"], "limit": 12}, max_items=12)
            for e in entries or []:
                if not _within(e.get("published"), cutoff):
                    continue
                found.append({
                    "source": "blog",
                    "source_id": e.get("url", ""),
                    "title": e.get("title", ""),
                    "abstract": e.get("summary", ""),
                    "url": e.get("url", ""),
                    "published": e.get("published", ""),
                    "authors": [e.get("author", "")] if e.get("author") else [],
                    "venue": feed.get("name", e.get("feed_title", "")),
                })
        return found[:max_items]


class ConferenceAgent(SourceAgent):
    name = "conference_agent"
    source_name = "openreview"
    item_role = "paper"
    uses_skills = ("research_discovery", "paper_analysis", "claim_extraction")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        discovery = self.skill("research_discovery")
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        for topic in _rotate(get_config().get("research.topics", []), per_cycle=4):
            for tool_name, args in (
                ("search_openreview", {"query": topic, "limit": 8}),
                ("search_acl_anthology", {"query": topic, "limit": 6}),
            ):
                items = await discovery(self.ctx, tool=tool_name, tool_args=args,
                                        max_items=8)
                for it in items or []:
                    sid = str(it.get("source_id", ""))
                    if sid and sid not in seen:
                        seen.add(sid)
                        found.append(it)
        return found[:max_items]


class BenchmarkAgent(SourceAgent):
    name = "benchmark_agent"
    source_name = "github"
    item_role = "benchmark"
    uses_skills = ("research_discovery", "repo_analysis")

    async def discover(self, *, since_days: int, max_items: int) -> list[dict[str, Any]]:
        watch = get_config().source("benchmarks.watch") or []
        found: list[dict[str, Any]] = []

        for entry in watch:
            repo = entry.get("repo")
            if repo and "fetch_github_repo" in self.ctx.allowed_tools:
                info = await self.ctx.try_call("fetch_github_repo", default=None,
                                               full_name=repo, with_readme=True)
                if info and _recently_active(info.get("pushed_at"), max(since_days, 120)):
                    info["benchmark_name"] = entry.get("name")
                    info["benchmark_kind"] = entry.get("kind")
                    found.append(info)

        # New benchmarks matter as much as movement on existing ones: a saturated
        # benchmark being replaced is exactly the kind of change an evaluation
        # chapter goes stale over.
        for query in ("new llm benchmark", "evaluation benchmark language model"):
            items = await self.ctx.try_call(
                "search_github", default=[], query=query, limit=6,
                min_stars=200, pushed_days=max(since_days, 90))
            found.extend(items or [])
        return found[:max_items]

    async def analyse(self, screened):  # type: ignore[override]
        skill = self.skill("repo_analysis")
        out = []
        for item, screen in screened:
            result = await skill(self.ctx, item=item, screen=screen)
            if result:
                # Tag so the synthesiser counts this as benchmark evidence rather than
                # as one more implementation.
                result.raw["benchmark_name"] = (
                    item.get("benchmark_name") or item.get("title", ""))
                result.raw["benchmark_kind"] = item.get("benchmark_kind", "")
                out.append(result)
        return out


class CommunityAgent(SourceAgent):
    name = "community_agent"
    source_name = "hackernews"
    item_role = "community"
    uses_skills = ("research_discovery", "community_signal_analysis")

    async def execute(self, *, since_days: int | None = None,
                      max_items: int | None = None, **_: Any) -> list[ResearchItem]:
        """Community signal is aggregated per topic, not per thread.

        One thread is noise; a recurring theme across several threads is signal. The
        analysis skill therefore consumes a topic's threads together rather than
        producing a research item per post.
        """
        cfg = get_config()
        since_days = since_days or int(cfg.get("research.lookback_days", 21))
        min_points = int(cfg.source("community.hn_min_points") or 30)
        queries = _rotate(cfg.source("community.hn_queries") or [], per_cycle=5)
        skill = self.skill("community_signal_analysis")

        items: list[ResearchItem] = []
        for topic in queries:
            threads = await self.ctx.try_call(
                "search_hackernews", default=[], query=topic, limit=12,
                min_points=min_points, days=max(since_days, 90))
            discussions = await self.ctx.try_call(
                "search_github_discussions", default=[], query=topic, limit=8,
                days=max(since_days, 90))
            combined = (threads or []) + (discussions or [])
            if len(combined) < 2:
                continue  # a single thread is not a signal
            item = await skill(self.ctx, items=combined, topic=topic)
            if item:
                items.append(item)

        fresh = self._drop_seen_items(items)
        await self.persist(fresh)
        self.log.info(f"{self.name}: {len(fresh)} community signals recorded", status="ok")
        return fresh

    def _drop_seen_items(self, items: list[ResearchItem]) -> list[ResearchItem]:
        out = []
        for item in items:
            existing = self.store.query_one(
                "SELECT id FROM research_items WHERE source = ? AND source_id = ?",
                (item.source.source, item.source.source_id))
            if existing:
                self.store.execute(
                    "UPDATE research_items SET last_seen_at = ? WHERE id = ?",
                    (utcnow(), existing["id"]))
            else:
                out.append(item)
        return out

    async def discover(self, *, since_days: int, max_items: int):  # pragma: no cover
        return []


# -- helpers ---------------------------------------------------------------


def _rotate(values: list[str], *, per_cycle: int) -> list[str]:
    """Take a slice of a list that advances each day.

    Every topic gets covered over a few days without issuing one query per topic per
    cycle, which would exhaust rate-limited sources on the first agent.
    """
    if not values:
        return []
    if len(values) <= per_cycle:
        return list(values)
    day = datetime.now(timezone.utc).toordinal()
    start = (day * per_cycle) % len(values)
    doubled = list(values) + list(values)
    return doubled[start:start + per_cycle]


def _within(published: str | None, cutoff: datetime) -> bool:
    if not published:
        return True
    for parser in (
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        lambda s: datetime.strptime(s[:25], "%a, %d %b %Y %H:%M:%S").replace(
            tzinfo=timezone.utc),
    ):
        try:
            dt = parser(published)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff
        except (ValueError, TypeError):
            continue
    return True


def _recently_active(pushed_at: str | None, days: int) -> bool:
    if not pushed_at:
        return False
    try:
        dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)
