"""Generic web tools: search, page fetch, RSS.

``web_search`` has a pluggable backend. If any commercial key is configured it is used;
otherwise the system falls back to scraping DuckDuckGo's HTML endpoint, which the audit
found works via **POST** (a GET returns a page with no results). That fallback is
intentionally the last resort — it is the least reliable tool in the system, so callers
treat a failure as degradation rather than an error.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..config import get_config, get_secrets
from ..obs import get_logger
from .http import get_bytes, get_json, get_text, request
from .registry import Capability, ToolError, ToolUnavailable, tool

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@tool("web_search", [Capability.SEARCH],
      description="General web search via the best configured backend.")
async def web_search(query: str, *, limit: int = 8) -> list[dict[str, str]]:
    secrets = get_secrets()
    log = get_logger()

    backends: list[tuple[str, Any]] = []
    if secrets.has("BRAVE_SEARCH_API_KEY"):
        backends.append(("brave", _brave))
    if secrets.has("SERPER_API_KEY"):
        backends.append(("serper", _serper))
    if secrets.has("TAVILY_API_KEY"):
        backends.append(("tavily", _tavily))
    backends.append(("duckduckgo", _duckduckgo))
    backends.append(("startpage", _startpage))

    errors: list[str] = []
    for name, fn in backends:
        try:
            hits = await fn(query, limit)
            if hits:
                return [h.as_dict() for h in hits[:limit]]
            errors.append(f"{name}: no results")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            log.debug(f"web_search backend {name} failed: {exc}")

    raise ToolUnavailable("all web search backends failed: " + " | ".join(errors[:3]))


async def _brave(query: str, limit: int) -> list[SearchHit]:
    data = await get_json(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(limit, 20)},
        headers={"X-Subscription-Token": get_secrets().require("BRAVE_SEARCH_API_KEY"),
                 "Accept": "application/json"},
    )
    return [
        SearchHit(r.get("title", ""), r.get("url", ""),
                  _text_of(r.get("description", "")))
        for r in (data.get("web", {}).get("results", []) or [])
    ]


async def _serper(query: str, limit: int) -> list[SearchHit]:
    data = await get_json(
        "https://google.serper.dev/search",
        json_body={"q": query, "num": min(limit, 20)},
        headers={"X-API-KEY": get_secrets().require("SERPER_API_KEY"),
                 "Content-Type": "application/json"},
    )
    return [
        SearchHit(r.get("title", ""), r.get("link", ""), r.get("snippet", ""))
        for r in (data.get("organic", []) or [])
    ]


async def _tavily(query: str, limit: int) -> list[SearchHit]:
    data = await get_json(
        "https://api.tavily.com/search",
        json_body={"api_key": get_secrets().require("TAVILY_API_KEY"),
                   "query": query, "max_results": min(limit, 20)},
        headers={"Content-Type": "application/json"},
    )
    return [
        SearchHit(r.get("title", ""), r.get("url", ""), r.get("content", "")[:300])
        for r in (data.get("results", []) or [])
    ]


async def _duckduckgo(query: str, limit: int) -> list[SearchHit]:
    """DuckDuckGo HTML endpoint.

    Must be POST: the audit measured that a GET to this endpoint returns a page with
    zero result anchors, while the POST form returns real results.
    """
    resp = await request(
        "POST", "https://html.duckduckgo.com/html/",
        data={"q": query, "b": ""},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        browser_ua=True, timeout=30,
    )
    if resp.status_code != 200:
        raise ToolError(f"duckduckgo returned HTTP {resp.status_code}")
    return _parse_ddg(resp.text)[:limit]


def _parse_ddg(page: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    blocks = re.split(r'<div class="result[^"]*results_links', page)
    for block in blocks[1:]:
        m = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                      block, re.S)
        if not m:
            continue
        url = _unwrap_ddg(html.unescape(m.group(1)))
        title = _text_of(m.group(2))
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        snippet = _text_of(sm.group(1)) if sm else ""
        if url and title:
            hits.append(SearchHit(title, url, snippet))
    return hits


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results in /l/?uddg=<encoded>."""
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        if qs.get("uddg"):
            return unquote(qs["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


async def _startpage(query: str, limit: int) -> list[SearchHit]:
    page = await get_text(
        "https://www.startpage.com/sp/search", params={"query": query},
        browser_ua=True, timeout=30,
    )
    hits: list[SearchHit] = []
    for m in re.finditer(
        r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        page, re.S,
    ):
        url, title = html.unescape(m.group(1)), _text_of(m.group(2))
        if url.startswith("http") and title:
            hits.append(SearchHit(title, url, ""))
        if len(hits) >= limit:
            break
    return hits


@tool("fetch_url", [Capability.FETCH],
      description="Fetch a URL and return readable text plus metadata.")
async def fetch_url(
    url: str, *, max_chars: int = 60_000, raw: bool = False,
) -> dict[str, Any]:
    resp = await request("GET", url, browser_ua=True, timeout=45)
    if resp.status_code >= 400:
        raise ToolError(f"{url} -> HTTP {resp.status_code}")

    content_type = resp.headers.get("content-type", "")
    if "pdf" in content_type.lower():
        return {"url": str(resp.url), "content_type": content_type,
                "is_pdf": True, "bytes": len(resp.content), "text": ""}

    body = resp.text
    if raw:
        return {"url": str(resp.url), "content_type": content_type, "text": body[:max_chars]}

    return {
        "url": str(resp.url),
        "content_type": content_type,
        "status": resp.status_code,
        "title": _extract_title(body),
        "text": extract_readable_text(body)[:max_chars],
        "links": _extract_links(body, str(resp.url))[:60],
    }


def extract_readable_text(page: str) -> str:
    """Strip a page down to prose. Not a full reader, but enough to judge relevance."""
    text = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>",
                  " ", page)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?i)</(p|div|section|article|li|h[1-6]|tr|br)>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_title(page: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    return _text_of(m.group(1)) if m else ""


def _extract_links(page: str, base: str) -> list[dict[str, str]]:
    from urllib.parse import urljoin
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', page, re.S | re.I):
        href = urljoin(base, html.unescape(m.group(1)))
        if href in seen or not href.startswith("http"):
            continue
        seen.add(href)
        out.append({"url": href, "text": _text_of(m.group(2))[:120]})
    return out


@tool("fetch_rss", [Capability.FETCH],
      description="Fetch and parse an RSS/Atom feed.")
async def fetch_rss(url: str, *, limit: int = 25) -> list[dict[str, Any]]:
    raw = await get_bytes(url, timeout=45, max_bytes=8 * 1024 * 1024)
    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover
        raise ToolUnavailable("feedparser is not installed") from exc

    parsed = feedparser.parse(raw)
    out: list[dict[str, Any]] = []
    for entry in parsed.entries[:limit]:
        summary = entry.get("summary", "") or ""
        if entry.get("content"):
            summary = entry["content"][0].get("value", summary)
        out.append({
            "title": _text_of(entry.get("title", "")),
            "url": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "")),
            "author": entry.get("author", ""),
            "summary": extract_readable_text(summary)[:4000],
            "tags": [t.get("term", "") for t in entry.get("tags", []) or []],
            "feed_title": _text_of(parsed.feed.get("title", "")) if parsed.feed else "",
        })
    return out


def _text_of(fragment: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", fragment or ""))).strip()
