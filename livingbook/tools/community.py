"""Community signal tools.

Everything here produces *community* evidence with, at best, ``moderate`` strength.
A thread never becomes scientific evidence — that constraint is enforced in the
research models, not left to a prompt.

Reddit is implemented but disabled by default: it returns 403 to script user-agents
(measured). Hacker News via the Algolia API and GitHub issues/discussions carry the
load instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import get_config
from .http import get_json
from .registry import Capability, ToolError, ToolUnavailable, tool

HN_API = "https://hn.algolia.com/api/v1"


@tool("search_hackernews", [Capability.SEARCH],
      description="Search Hacker News stories and comments for practitioner discussion.")
async def search_hackernews(
    query: str, *, limit: int = 20, min_points: int = 20, days: int | None = 180,
    include_comments: bool = False,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "query": query,
        "tags": "(story,show_hn)" if not include_comments else "(story,comment)",
        "hitsPerPage": min(limit * 2, 100),
    }
    filters = []
    if min_points:
        filters.append(f"points>={min_points}")
    if days:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        filters.append(f"created_at_i>{cutoff}")
    if filters:
        params["numericFilters"] = ",".join(filters)

    data = await get_json(f"{HN_API}/search", params=params, timeout=45)
    out: list[dict[str, Any]] = []
    for hit in data.get("hits", []):
        title = hit.get("title") or hit.get("story_title") or ""
        if not title:
            continue
        out.append({
            "source": "hackernews",
            "source_id": str(hit.get("objectID", "")),
            "title": title,
            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "discussion_url": f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "points": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "author": hit.get("author", ""),
            "published": hit.get("created_at", ""),
            "text": (hit.get("story_text") or hit.get("comment_text") or "")[:4000],
        })
    out.sort(key=lambda d: -(d["points"] + d["num_comments"]))
    return out[:limit]


@tool("fetch_hn_thread", [Capability.FETCH],
      description="Fetch a Hacker News thread with its top comments.")
async def fetch_hn_thread(item_id: str, *, max_comments: int = 40) -> dict[str, Any]:
    data = await get_json(f"{HN_API}/items/{item_id}", timeout=45)
    comments: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], depth: int = 0) -> None:
        if len(comments) >= max_comments:
            return
        for child in node.get("children", []) or []:
            text = child.get("text") or ""
            if text:
                comments.append({
                    "author": child.get("author", ""), "text": text[:2500],
                    "points": child.get("points") or 0, "depth": depth,
                })
            walk(child, depth + 1)

    walk(data)
    return {
        "source_id": str(item_id),
        "title": data.get("title", ""),
        "url": data.get("url", ""),
        "points": data.get("points", 0),
        "author": data.get("author", ""),
        "published": data.get("created_at", ""),
        "text": (data.get("text") or "")[:6000],
        "comments": comments,
    }


@tool("search_reddit", [Capability.SEARCH],
      description="Search subreddits. Disabled by default: Reddit blocks script clients.")
async def search_reddit(
    query: str, *, subreddits: list[str] | None = None, limit: int = 15, sort: str = "relevance",
) -> list[dict[str, Any]]:
    """Measured 403 from reddit.com and old.reddit.com for non-browser clients.

    Kept as a config flag rather than deleted so it can be switched on without a code
    change if credentials or a proxy become available.
    """
    cfg = get_config()
    if not cfg.get("research.sources.reddit.enabled", False):
        raise ToolUnavailable(
            "Reddit is disabled: it returns HTTP 403 to script user-agents. "
            "Enable research.sources.reddit.enabled in config/config.yaml to retry."
        )

    subs = subreddits or (cfg.source("community.reddit.subreddits") or ["MachineLearning"])
    out: list[dict[str, Any]] = []
    for sub in subs[:4]:
        try:
            data = await get_json(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": query, "restrict_sr": 1, "sort": sort,
                        "limit": min(limit, 25), "t": "year"},
                headers={"User-Agent": "python:livingbook:0.1 (research bot)"},
                timeout=45,
            )
        except (ToolError, ToolUnavailable):
            continue
        for child in (data.get("data", {}) or {}).get("children", []):
            post = child.get("data", {}) or {}
            out.append({
                "source": "reddit",
                "source_id": post.get("id", ""),
                "title": post.get("title", ""),
                "url": f"https://reddit.com{post.get('permalink','')}",
                "subreddit": post.get("subreddit", sub),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "text": (post.get("selftext") or "")[:4000],
                "published": datetime.fromtimestamp(
                    post.get("created_utc", 0), tz=timezone.utc).isoformat()
                if post.get("created_utc") else "",
            })
    out.sort(key=lambda d: -(d["score"] + d["num_comments"]))
    return out[:limit]
