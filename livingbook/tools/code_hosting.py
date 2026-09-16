"""GitHub and Hugging Face tools — implementation, release and adoption signals.

Everything here produces *practical* or *adoption* evidence, never scientific evidence.
A repository with 20k stars tells you a technique is used, not that it works better;
that distinction is enforced at the schema level in the research models.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import get_config, get_secrets
from .http import get_json, github_headers, request
from .registry import Capability, ToolError, ToolUnavailable, tool

GITHUB_API = "https://api.github.com"
HF_API = "https://huggingface.co/api"
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


# ──────────────────────────────── GitHub ─────────────────────────────────


@tool("search_github", [Capability.SEARCH],
      description="Search GitHub repositories by topic and activity.")
async def search_github(
    query: str, *, limit: int = 15, min_stars: int = 0, pushed_days: int | None = None,
) -> list[dict[str, Any]]:
    q = query
    if min_stars:
        q += f" stars:>={min_stars}"
    if pushed_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=pushed_days)).date()
        q += f" pushed:>={cutoff.isoformat()}"

    data = await get_json(
        f"{GITHUB_API}/search/repositories",
        params={"q": q, "sort": "stars", "order": "desc", "per_page": min(limit, 50)},
        headers=github_headers(), timeout=45,
    )
    return [_normalise_repo(r) for r in data.get("items", [])]


@tool("fetch_github_repo", [Capability.FETCH],
      description="Fetch repository metadata, README and paper links.")
async def fetch_github_repo(full_name: str, *, with_readme: bool = True) -> dict[str, Any]:
    repo = await get_json(f"{GITHUB_API}/repos/{full_name}",
                          headers=github_headers(), timeout=45)
    out = _normalise_repo(repo)

    if with_readme:
        try:
            readme = await get_json(f"{GITHUB_API}/repos/{full_name}/readme",
                                    headers=github_headers(), timeout=45)
            content = base64.b64decode(readme.get("content", "")).decode("utf-8", "replace")
            out["readme"] = content[:40_000]
            # Linking a repo to its paper is what lets the synthesiser fuse an
            # implementation signal with the scientific evidence it implements.
            out["arxiv_ids"] = sorted(set(_ARXIV_RE.findall(content)))
        except (ToolError, ToolUnavailable):
            out["readme"] = ""
            out["arxiv_ids"] = []
    return out


@tool("fetch_github_activity", [Capability.FETCH],
      description="Fetch releases, commit cadence and issue activity for a repo.")
async def fetch_github_activity(full_name: str, *, days: int = 90) -> dict[str, Any]:
    """Maintenance and adoption signal: is this alive, and is it moving?"""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    out: dict[str, Any] = {"full_name": full_name, "window_days": days}

    try:
        releases = await get_json(
            f"{GITHUB_API}/repos/{full_name}/releases",
            params={"per_page": 10}, headers=github_headers(), timeout=45)
        out["releases"] = [
            {"tag": r.get("tag_name"), "name": r.get("name"),
             "published_at": r.get("published_at"),
             "body": (r.get("body") or "")[:3000], "prerelease": r.get("prerelease")}
            for r in releases
        ]
    except (ToolError, ToolUnavailable):
        out["releases"] = []

    try:
        commits = await get_json(
            f"{GITHUB_API}/repos/{full_name}/commits",
            params={"since": since, "per_page": 100},
            headers=github_headers(), timeout=45)
        out["commits_in_window"] = len(commits)
        out["last_commit"] = (
            commits[0].get("commit", {}).get("author", {}).get("date") if commits else None
        )
        out["recent_commit_messages"] = [
            (c.get("commit", {}).get("message") or "").split("\n")[0][:140]
            for c in commits[:15]
        ]
    except (ToolError, ToolUnavailable):
        out["commits_in_window"] = None

    try:
        issues = await get_json(
            f"{GITHUB_API}/search/issues",
            params={"q": f"repo:{full_name} is:issue created:>={since[:10]}",
                    "per_page": 1},
            headers=github_headers(), timeout=45)
        out["issues_opened_in_window"] = issues.get("total_count", 0)
    except (ToolError, ToolUnavailable):
        out["issues_opened_in_window"] = None

    return out


def _normalise_repo(repo: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "github",
        "source_id": repo.get("full_name", ""),
        "title": repo.get("full_name", ""),
        "description": repo.get("description") or "",
        "url": repo.get("html_url", ""),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "watchers": repo.get("subscribers_count", repo.get("watchers_count", 0)),
        "open_issues": repo.get("open_issues_count", 0),
        "language": repo.get("language"),
        "topics": repo.get("topics", []),
        "license": ((repo.get("license") or {}).get("spdx_id")
                    if repo.get("license") else None),
        "created_at": repo.get("created_at"),
        "pushed_at": repo.get("pushed_at"),
        "archived": repo.get("archived", False),
        "homepage": repo.get("homepage") or "",
    }


@tool("search_github_discussions", [Capability.SEARCH],
      description="Search issues and discussions in a repo — practitioner pain points.")
async def search_github_discussions(
    query: str, *, repos: list[str] | None = None, limit: int = 15, days: int = 120,
) -> list[dict[str, Any]]:
    repos = repos or (get_config().source("community.github_discussion_repos") or [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    out: list[dict[str, Any]] = []

    for repo in repos[:6]:
        try:
            data = await get_json(
                f"{GITHUB_API}/search/issues",
                params={"q": f"repo:{repo} {query} created:>={cutoff}",
                        "sort": "reactions", "order": "desc",
                        "per_page": min(limit, 20)},
                headers=github_headers(), timeout=45)
        except (ToolError, ToolUnavailable):
            continue
        for item in data.get("items", []):
            out.append({
                "source": "github_discussion",
                "source_id": str(item.get("id", "")),
                "title": item.get("title", ""),
                "url": item.get("html_url", ""),
                "body": (item.get("body") or "")[:4000],
                "repo": repo,
                "state": item.get("state"),
                "comments": item.get("comments", 0),
                "reactions": (item.get("reactions") or {}).get("total_count", 0),
                "published": item.get("created_at"),
                "labels": [l.get("name") for l in (item.get("labels") or [])],
            })
    out.sort(key=lambda d: -(d["reactions"] + d["comments"]))
    return out[:limit]


# ───────────────────────────── Hugging Face ──────────────────────────────


@tool("search_huggingface", [Capability.SEARCH],
      description="Search Hugging Face models or datasets.")
async def search_huggingface(
    query: str, *, kind: str = "models", limit: int = 15, sort: str = "downloads",
) -> list[dict[str, Any]]:
    if kind not in ("models", "datasets"):
        raise ToolError(f"kind must be 'models' or 'datasets', got {kind!r}")
    data = await get_json(
        f"{HF_API}/{kind}",
        params={"search": query, "sort": sort, "direction": -1,
                "limit": min(limit, 50), "full": "true"},
        timeout=45,
    )
    return [_normalise_hf(item, kind) for item in (data or [])]


@tool("fetch_hf_model", [Capability.FETCH],
      description="Fetch a Hugging Face model card and metadata.")
async def fetch_hf_model(model_id: str) -> dict[str, Any]:
    data = await get_json(f"{HF_API}/models/{model_id}", timeout=45)
    out = _normalise_hf(data, "models")
    try:
        card = await request(
            "GET", f"https://huggingface.co/{model_id}/raw/main/README.md", timeout=45)
        if card.status_code == 200:
            out["model_card"] = card.text[:40_000]
            out["arxiv_ids"] = sorted(set(_ARXIV_RE.findall(card.text)))
    except (ToolError, ToolUnavailable):
        out["model_card"] = ""
    return out


@tool("search_hf_papers", [Capability.SEARCH],
      description="Fetch Hugging Face daily papers — a curated community-attention signal.")
async def search_hf_papers(*, limit: int = 30, days: int | None = None) -> list[dict[str, Any]]:
    data = await get_json(f"{HF_API}/daily_papers",
                          params={"limit": min(limit, 100)}, timeout=45)
    out: list[dict[str, Any]] = []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)) if days else None

    for item in (data or []):
        paper = item.get("paper", {}) or {}
        published = item.get("publishedAt") or paper.get("publishedAt") or ""
        if cutoff and published:
            try:
                if datetime.fromisoformat(published.replace("Z", "+00:00")) < cutoff:
                    continue
            except ValueError:
                pass
        out.append({
            "source": "hf_papers",
            "source_id": paper.get("id", ""),
            "arxiv_id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("summary", "")[:6000],
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
            "upvotes": paper.get("upvotes", 0),
            "comments": item.get("numComments", 0),
            "published": published,
            "url": f"https://huggingface.co/papers/{paper.get('id','')}",
        })
    return out


def _normalise_hf(item: dict[str, Any], kind: str) -> dict[str, Any]:
    ident = item.get("id", item.get("modelId", ""))
    card = item.get("cardData") or {}
    return {
        "source": "huggingface",
        "source_id": ident,
        "title": ident,
        "kind": kind,
        "url": f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}{ident}",
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
        "tags": item.get("tags", []),
        "pipeline_tag": item.get("pipeline_tag"),
        "library": item.get("library_name"),
        "license": card.get("license") or _license_from_tags(item.get("tags", [])),
        "created_at": item.get("createdAt"),
        "last_modified": item.get("lastModified"),
        "author": item.get("author") or ident.split("/")[0] if "/" in ident else "",
    }


def _license_from_tags(tags: list[str]) -> str | None:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    return None
