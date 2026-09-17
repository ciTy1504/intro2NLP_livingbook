"""Dashboard data export.

Reads the live SQLite state and produces a single JSON payload. The HTML page embeds
it, so the result is one self-contained file: it opens from disk with no server, it
deploys to Firebase Hosting as a static asset, and it publishes as an artifact —
all from the same generator, with no runtime backend anywhere.

That matters because the alternative, a small API in front of the database, would need
hosting, and everything this system operates on is local.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_config
from .state.store import get_store


def _rows(store, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in store.query(sql, params)]


def collect() -> dict[str, Any]:
    """Everything the dashboard shows, in one pass over the database."""
    cfg = get_config()
    store = get_store()
    scalar = store.scalar

    verdicts = {r["decision"]: r["n"] for r in store.query(
        "SELECT decision, COUNT(*) n FROM verdicts GROUP BY decision")}
    changed = sum(n for d, n in verdicts.items()
                  if d not in ("MONITOR", "IGNORE"))
    refused = sum(n for d, n in verdicts.items()
                  if d in ("MONITOR", "IGNORE"))

    # Evidence kinds, ordered by how much weight the system gives them. The order is
    # the point: it is the signal-versus-evidence distinction made visible.
    kind_order = ["scientific", "independent_verification", "practical",
                  "adoption", "author_claim", "community", "anecdotal_practical"]
    evidence_raw = {r["kind"]: r["n"] for r in store.query(
        "SELECT kind, COUNT(*) n FROM evidence GROUP BY kind")}
    evidence = [{"kind": k, "count": evidence_raw.get(k, 0)}
                for k in kind_order if evidence_raw.get(k)]

    maturity_order = ["speculative", "emerging", "consolidating", "mature", "superseded"]
    maturity_raw = {r["maturity"]: r["n"] for r in store.query(
        "SELECT maturity, COUNT(*) n FROM clusters GROUP BY maturity")}
    maturity = [{"level": m, "count": maturity_raw.get(m, 0)}
                for m in maturity_order if maturity_raw.get(m)]

    stuck = _rows(store,
        "SELECT p.id, p.state, p.revisions, p.last_error, p.updated_at, "
        "c.title, c.maturity FROM pipelines p LEFT JOIN clusters c ON c.id = p.cluster_id "
        "WHERE p.state IN ('FAILED','NEEDS_HUMAN') ORDER BY p.updated_at DESC")

    accepted_changes = _rows(store,
        "SELECT v.decision, v.scope, v.rationale, c.title, c.maturity "
        "FROM verdicts v JOIN clusters c ON c.id = v.cluster_id "
        "WHERE v.decision NOT IN ('MONITOR','IGNORE') ORDER BY v.created_at DESC LIMIT 10")

    refusals = _rows(store,
        "SELECT c.title, c.maturity, v.rationale FROM verdicts v "
        "JOIN clusters c ON c.id = v.cluster_id WHERE v.decision = 'MONITOR' "
        "ORDER BY v.created_at DESC LIMIT 8")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "book": {
            "title": cfg.get("project.book_title", ""),
            # Repo name only, not a URL: the page carries no outbound links.
            "repo": str(cfg.get("project.repo_url", "")).rstrip("/").rsplit("/", 1)[-1],
            "nodes": scalar("SELECT COUNT(*) FROM book_nodes") or 0,
            "chapters": scalar("SELECT COUNT(*) FROM book_nodes WHERE kind='chapter'") or 0,
            "sections": scalar("SELECT COUNT(*) FROM book_nodes WHERE kind='section'") or 0,
            # Every node's body stops at the next heading of any level, so node bodies do
            # not nest and summing them all gives the book's real word count.
            "words": scalar("SELECT SUM(word_count) FROM book_nodes") or 0,
            "concepts": scalar("SELECT COUNT(*) FROM concepts") or 0,
            "claims": scalar("SELECT COUNT(*) FROM claims") or 0,
            "claims_needing_citation": scalar(
                "SELECT COUNT(*) FROM claims WHERE needs_citation=1") or 0,
            "bib_entries": scalar("SELECT COUNT(*) FROM bib_entries") or 0,
            "figures": scalar("SELECT COUNT(*) FROM figures") or 0,
            "graph_edges": scalar("SELECT COUNT(*) FROM graph_edges") or 0,
        },
        "research": {
            "total": scalar("SELECT COUNT(*) FROM research_items") or 0,
            "by_source": _rows(store,
                "SELECT source, COUNT(*) n FROM research_items "
                "GROUP BY source ORDER BY n DESC"),
            "by_lifecycle": _rows(store,
                "SELECT lifecycle, COUNT(*) n FROM research_items GROUP BY lifecycle"),
            "evidence_total": scalar("SELECT COUNT(*) FROM evidence") or 0,
            "evidence": evidence,
        },
        "clusters": {
            "total": scalar("SELECT COUNT(*) FROM clusters") or 0,
            "maturity": maturity,
        },
        "verdicts": {
            "counts": verdicts,
            "changed": changed,
            "refused": refused,
            "accepted": accepted_changes,
            "refusals": refusals,
        },
        "pipelines": {
            "by_state": _rows(store,
                "SELECT state, COUNT(*) n FROM pipelines GROUP BY state ORDER BY n DESC"),
            "stuck": stuck,
        },
        # No URLs in the payload: the page is a read-only snapshot, and drill-down
        # belongs to the CLI and the MCP tools, which carry the whole provenance
        # chain rather than a bare link.
        "recent_research": _rows(store,
            "SELECT source, title, published_at, lifecycle, relevance "
            "FROM research_items ORDER BY discovered_at DESC LIMIT 25"),
        "scheduler": _rows(store,
            "SELECT name, interval_seconds, last_run_at, next_run_at, last_status, runs "
            "FROM scheduled_jobs ORDER BY next_run_at"),
        "llm": _llm_stats(cfg),
        "activity": _rows(store,
            "SELECT ts, level, agent, message FROM events "
            "WHERE level IN ('INFO','WARN','ERROR') AND agent IS NOT NULL "
            "ORDER BY id DESC LIMIT 40"),
        "totals": {
            "events": scalar("SELECT COUNT(*) FROM events") or 0,
            "artifacts": scalar("SELECT COUNT(*) FROM artifacts") or 0,
            "runs": scalar("SELECT COUNT(*) FROM runs") or 0,
        },
    }


def _llm_stats(cfg) -> dict[str, Any]:
    path = cfg.root / "state" / "keypool_stats.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    requests = sum(v.get("requests", 0) for v in data.values())
    successes = sum(v.get("successes", 0) for v in data.values())
    return {
        "keys": len(data),
        "disabled": sum(1 for v in data.values() if v.get("disabled")),
        "requests": requests,
        "successes": successes,
        "success_rate": round(100 * successes / requests, 1) if requests else 0.0,
        "tokens_in": sum(v.get("tokens_in", 0) for v in data.values()),
        "tokens_out": sum(v.get("tokens_out", 0) for v in data.values()),
    }


def render(template: Path, data: dict[str, Any]) -> str:
    """Inject the payload into the page's data island."""
    html = template.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    marker_open = '<script id="livingbook-data" type="application/json">'
    marker_close = "</script>"
    start = html.index(marker_open) + len(marker_open)
    end = html.index(marker_close, start)
    # </script> inside JSON would terminate the island early.
    return html[:start] + payload.replace("</", "<\\/") + html[end:]


def build(output: Path | None = None) -> Path:
    cfg = get_config()
    template = cfg.root / "dashboard" / "template.html"
    if not template.exists():
        raise FileNotFoundError(f"dashboard template missing: {template}")
    out = output or (cfg.root / "dashboard" / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(template, collect()), encoding="utf-8")
    return out
