"""Measure the real pairwise-similarity distribution of research items.

    python scripts/calibrate_clustering.py

Run this after changing the embedding model, and whenever clusters look wrong.

The synthesis thresholds cannot be reasoned about from intuition. Embeddings of any two
machine-learning abstracts sit at a high baseline cosine — measured here at min 0.749
across 780 real pairs — so a threshold picked for "these texts are similar" merges
everything into one cluster. The original 0.74 was below the minimum observed pair.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.llm import get_provider
from livingbook.state.store import get_store


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na and nb else 0.0


async def main():
    store = get_store()
    rows = store.query(
        "SELECT id, title, summary, concepts_json FROM research_items LIMIT 40")
    items = []
    for r in rows:
        try:
            concepts = json.loads(r["concepts_json"] or "[]")
        except Exception:
            concepts = []
        items.append({"id": r["id"], "title": r["title"],
                      "text": f"{r['title']}. {r['summary']} {' '.join(concepts)}"[:2000],
                      "concepts": {c.lower() for c in concepts}})
    print(f"{len(items)} research items")

    resp = await get_provider().embed([i["text"] for i in items])
    vecs = resp.vectors

    sims, overlaps = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sims.append(cos(vecs[i], vecs[j]))
            ci, cj = items[i]["concepts"], items[j]["concepts"]
            overlaps.append(len(ci & cj) / max(1, min(len(ci), len(cj))) if ci and cj else 0.0)

    sims.sort()
    print("\nPairwise EMBEDDING cosine across all pairs:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  p{int(q*100):<3d} {sims[int(q * (len(sims) - 1))]:.3f}")
    print(f"  mean {statistics.mean(sims):.3f}  max {max(sims):.3f}  min {min(sims):.3f}")

    overlaps.sort()
    nonzero = [o for o in overlaps if o > 0]
    print("\nPairwise CONCEPT overlap:")
    print(f"  pairs with any overlap: {len(nonzero)}/{len(overlaps)}")
    if nonzero:
        for q in (0.5, 0.75, 0.9, 0.99):
            print(f"  p{int(q*100):<3d} (of non-zero) {nonzero[int(q*(len(nonzero)-1))]:.3f}")

    print("\nHow many clusters would each threshold produce (mean-affinity assignment)?")
    for thr in (0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90):
        clusters = []
        for i in range(len(items)):
            best, best_s = None, 0.0
            for c in clusters:
                s = sum(max(cos(vecs[i], vecs[j]),
                            len(items[i]["concepts"] & items[j]["concepts"]) /
                            max(1, min(len(items[i]["concepts"]), len(items[j]["concepts"])))
                            if items[i]["concepts"] and items[j]["concepts"] else 0.0)
                        for j in c) / len(c)
                if s > best_s:
                    best, best_s = c, s
            if best is not None and best_s >= thr:
                best.append(i)
            else:
                clusters.append([i])
        sizes = sorted((len(c) for c in clusters), reverse=True)
        print(f"  thr={thr:.2f} -> {len(clusters):2d} clusters, sizes {sizes[:8]}")

    await get_provider().aclose()


asyncio.run(main())
