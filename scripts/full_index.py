"""Build the complete Book Knowledge Base from the manuscript.

Run once at setup, and after any large manual edit. Incremental by content hash, so
re-running it after a small change costs a handful of calls rather than several hundred.

    python scripts/full_index.py [--force] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.knowledge.graph import KnowledgeGraph          # noqa: E402
from livingbook.knowledge.indexer import BookIndexer           # noqa: E402
from livingbook.llm import get_provider                        # noqa: E402
from livingbook.obs import get_logger, run_scope               # noqa: E402
from livingbook.state.store import get_store                   # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-summarise every node even if unchanged")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    log = get_logger()
    store = get_store()

    with run_scope() as run_id:
        store.start_run(run_id, trigger="full_index")
        indexer = BookIndexer()

        book, structural = indexer.index_structure()
        log.info(f"structure: {json.dumps(structural.as_dict())}")

        semantic = await indexer.index_semantics(
            book, force=args.force, concurrency=args.concurrency, limit=args.limit
        )
        log.info(f"semantics: {json.dumps(semantic.as_dict())}")

        graph = KnowledgeGraph(store)
        log.info(f"graph: {json.dumps(graph.stats())}")
        log.info(f"db: {json.dumps(store.counts())}")

        provider = get_provider()
        log.info(f"llm usage: {json.dumps(provider.status()['usage'])}")
        log.info(f"keypool: {json.dumps(provider.status()['keypool'])}")
        store.end_run(run_id, "completed", {
            "structural": structural.as_dict(), "semantic": semantic.as_dict(),
        })
        await provider.aclose()

    print("\nIndex complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
