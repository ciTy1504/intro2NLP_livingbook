"""Shared HTTP client for every external source.

Centralised so that politeness (User-Agent, contact email), per-host rate limiting,
retry and error classification are applied uniformly. The alternative — each tool
module making its own requests — is how a research bot ends up rate-limited out of a
source it depends on.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

from ..config import get_secrets
from ..obs import get_logger
from .registry import ToolError, ToolUnavailable

USER_AGENT = (
    "livingbook/0.1 (autonomous textbook research agent; "
    "+https://github.com/ciTy1504/intro2NLP_livingbook)"
)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

#: Minimum seconds between requests to the same host. Values reflect each source's
#: published guidance, or measured tolerance where none is published.
HOST_INTERVALS: dict[str, float] = {
    "export.arxiv.org": 3.0,           # arXiv asks for one request per 3s
    "api.semanticscholar.org": 3.0,    # hard 429s without a key
    "api.openalex.org": 0.15,
    "api.crossref.org": 0.2,
    "api.datacite.org": 0.3,
    "api.github.com": 0.8,
    "huggingface.co": 0.3,
    "hn.algolia.com": 0.3,
    "commons.wikimedia.org": 0.5,
    "api.openverse.org": 0.5,
    "api2.openreview.net": 0.5,
    "aclanthology.org": 1.0,
    "html.duckduckgo.com": 2.0,
    "www.startpage.com": 2.0,
}
DEFAULT_INTERVAL = 0.4


class HostLimiter:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def wait(self, host: str) -> None:
        interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
        async with self._lock(host):
            now = time.monotonic()
            elapsed = now - self._last.get(host, 0.0)
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
            self._last[host] = time.monotonic()


_limiter = HostLimiter()
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=15.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def request(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    json_body: Any = None,
    data: Any = None,
    timeout: float = 45.0,
    retries: int = 3,
    browser_ua: bool = False,
    polite: bool = True,
) -> httpx.Response:
    host = urlparse(url).netloc
    client = _get_client()
    log = get_logger()

    merged: dict[str, str] = {}
    if browser_ua:
        merged["User-Agent"] = BROWSER_UA
    if headers:
        merged.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        if polite:
            await _limiter.wait(host)
        try:
            resp = await client.request(
                method, url, params=params, headers=merged or None,
                json=json_body, data=data, timeout=httpx.Timeout(timeout, connect=15.0),
            )
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt == retries:
                raise ToolUnavailable(f"{host}: timed out after {retries} attempts") from exc
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == retries:
                raise ToolUnavailable(f"{host}: {type(exc).__name__}: {exc}") from exc
        else:
            if resp.status_code == 429:
                # Respect Retry-After when the server sends one; it is usually far more
                # accurate than our backoff guess.
                delay = _retry_after(resp) or min(8.0 * attempt, 30.0)
                log.debug(f"{host}: 429, backing off {delay:.1f}s (attempt {attempt})")
                if attempt == retries:
                    raise ToolUnavailable(f"{host}: rate limited after {retries} attempts")
                await asyncio.sleep(delay)
                continue
            if resp.status_code in (403, 451):
                raise ToolUnavailable(f"{host}: HTTP {resp.status_code} (blocked)")
            if 500 <= resp.status_code < 600:
                if attempt == retries:
                    raise ToolUnavailable(f"{host}: HTTP {resp.status_code}")
                await asyncio.sleep(min(2.0 * attempt, 10.0) * (0.5 + random.random()))
                continue
            return resp

        await asyncio.sleep(min(2.0 * attempt, 10.0) * (0.5 + random.random()))

    raise ToolUnavailable(f"{host}: exhausted retries ({last_exc})")


async def get_json(url: str, **kw: Any) -> Any:
    resp = await request("GET", url, **kw)
    if resp.status_code >= 400:
        raise ToolError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except Exception as exc:
        raise ToolError(f"{url} did not return JSON: {resp.text[:200]}") from exc


async def get_text(url: str, **kw: Any) -> str:
    resp = await request("GET", url, **kw)
    if resp.status_code >= 400:
        raise ToolError(f"{url} -> HTTP {resp.status_code}")
    return resp.text


async def get_bytes(url: str, *, max_bytes: int = 40 * 1024 * 1024, **kw: Any) -> bytes:
    resp = await request("GET", url, **kw)
    if resp.status_code >= 400:
        raise ToolError(f"{url} -> HTTP {resp.status_code}")
    content = resp.content
    if len(content) > max_bytes:
        raise ToolError(f"{url} exceeded {max_bytes} bytes ({len(content)})")
    return content


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def polite_params(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Params that identify us to sources with a polite pool (OpenAlex, Crossref)."""
    params: dict[str, Any] = {"mailto": get_secrets().contact_email}
    if extra:
        params.update(extra)
    return params


def github_headers() -> dict[str, str]:
    """Auth header when a token is configured — 60 req/h becomes 5000."""
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = get_secrets().get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
