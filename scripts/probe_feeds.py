"""Check every research blog feed in config/sources.yaml still resolves.

    python scripts/probe_feeds.py

Run this before adding a feed, and occasionally afterwards. Feeds rot, and a dead one
costs the discovery cycle ~12s of timeout every run while filling the log with warnings
that hide real problems. Add candidates to CANDIDATES below to test them before
committing them to the configuration.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

#: Feeds to evaluate but not yet configured.
CANDIDATES: list[dict[str, str]] = []


def probe(name: str, url: str) -> bool:
    started = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=12) as resp:
            head = resp.read(4096)
        ok = b"<rss" in head or b"<feed" in head or b"<?xml" in head
        status = "OK  " if ok else "ODD "
    except urllib.error.HTTPError as exc:
        ok, status = False, f"HTTP{exc.code}"
    except Exception as exc:
        ok, status = False, type(exc).__name__[:7]
    print(f"  {status:7s} {time.time() - started:4.1f}s  {name[:28]:30s} {url}")
    return ok


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
    feeds = cfg["blogs"]["feeds"]

    print(f"Configured feeds ({len(feeds)}):")
    failures = [f for f in feeds if not probe(f["name"], f["url"])]

    if CANDIDATES:
        print(f"\nCandidates ({len(CANDIDATES)}):")
        for f in CANDIDATES:
            probe(f["name"], f["url"])

    print()
    if failures:
        print(f"{len(failures)} configured feed(s) failed: "
              + ", ".join(f["name"] for f in failures))
        print("Remove or replace them in config/sources.yaml.")
        return 1
    print("All configured feeds resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
