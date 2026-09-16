"""Check every research blog feed in config/sources.yaml is actually fetchable.

    python scripts/probe_feeds.py

Uses the real ``fetch_rss`` tool rather than a separate HTTP call. An earlier version
made its own request and reported pytorch.org as healthy while the system itself got
403 from it — a probe that does not exercise the production path can only tell you
about a path nobody uses.

Feeds rot, and a dead one costs the discovery cycle a timeout every run while filling
the log with warnings that hide real problems. Add candidates to CANDIDATES to test
them before committing them to the configuration.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from livingbook.tools import system_context  # noqa: E402

#: Feeds to evaluate but not yet configured.
CANDIDATES: list[dict[str, str]] = []


async def probe(ctx, name: str, url: str) -> bool:
    started = time.time()
    try:
        entries = await ctx.call("fetch_rss", url=url, limit=3)
    except Exception as exc:
        print(f"  FAIL    {time.time() - started:5.1f}s  {name[:28]:30s} "
              f"{type(exc).__name__}: {str(exc)[:60]}")
        return False
    if not entries:
        print(f"  EMPTY   {time.time() - started:5.1f}s  {name[:28]:30s} {url}")
        return False
    print(f"  OK      {time.time() - started:5.1f}s  {name[:28]:30s} "
          f"{len(entries)} entries — {entries[0]['title'][:44]}")
    return True


async def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
    feeds = cfg["blogs"]["feeds"]
    ctx = system_context("probe")

    print(f"Configured feeds ({len(feeds)}):")
    results = [await probe(ctx, f["name"], f["url"]) for f in feeds]
    failures = [f for f, ok in zip(feeds, results) if not ok]

    if CANDIDATES:
        print(f"\nCandidates ({len(CANDIDATES)}):")
        for f in CANDIDATES:
            await probe(ctx, f["name"], f["url"])

    print()
    if failures:
        print(f"{len(failures)} configured feed(s) failed: "
              + ", ".join(f["name"] for f in failures))
        print("Remove or replace them in config/sources.yaml.")
        return 1
    print("All configured feeds fetch successfully through the production tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
