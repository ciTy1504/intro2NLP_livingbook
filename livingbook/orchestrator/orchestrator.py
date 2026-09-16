"""The orchestrator.

Schedules work, advances pipelines, enforces dependencies and approval gates. It has
no LLM grant at all — by contract, in ``config/agents.yaml``. It decides *when* things
run, never *whether the research is any good*; that judgement belongs to agents whose
output is verifiable and whose reasoning is recorded.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from ..agents import SOURCE_AGENTS, ResearchSynthesizer
from ..config import get_config
from ..knowledge.indexer import BookIndexer
from ..obs import get_logger, new_id, run_scope
from ..state.machine import Pipeline, State, StateMachine
from ..state.store import get_store, utcnow
from .pipeline import PipelineDriver
from .scheduler import Scheduler


class Orchestrator:
    def __init__(self) -> None:
        self.cfg = get_config()
        self.log = get_logger()
        self.store = get_store()
        self.machine = StateMachine(self.store)
        self.driver = PipelineDriver()
        self.scheduler = Scheduler(self.store)
        self._register_jobs()

    def _register_jobs(self) -> None:
        self.scheduler.register_from_config({
            "discovery": self.job_discovery,
            "synthesis": self.job_synthesis,
            "verdict": self.job_verdict,
            "pipeline_tick": self.job_pipeline_tick,
            "monitor_sweep": self.job_monitor_sweep,
            "global_qa": self.job_global_qa,
            "kb_reindex": self.job_kb_reindex,
        })

    # ── jobs ──────────────────────────────────────────────────────────────
    async def job_discovery(self) -> dict[str, Any]:
        """Fan out to the source agents with bounded concurrency."""
        limit = int(self.cfg.get("research.max_concurrent_source_agents", 4))
        sem = asyncio.Semaphore(limit)
        enabled = self._enabled_sources()

        async def run_one(cls: type) -> tuple[str, int]:
            agent = cls()
            if agent.name not in enabled:
                return agent.name, 0
            async with sem:
                try:
                    items = await agent.run()
                    return agent.name, len(items)
                except Exception as exc:
                    # Degrade, never abort: one dead source must not cost a cycle.
                    self.log.warn(
                        f"{agent.name} failed: {type(exc).__name__}: {exc}")
                    return agent.name, 0

        results = await asyncio.gather(*(run_one(c) for c in SOURCE_AGENTS))
        summary = {name: n for name, n in results}
        total = sum(summary.values())
        self.log.info(f"discovery complete: {total} items — {summary}", status="ok")
        return {"total": total, "by_agent": summary}

    async def job_synthesis(self) -> dict[str, Any]:
        clusters = await ResearchSynthesizer().run()
        created = 0
        for cluster in clusters:
            # One pipeline per cluster, and only one: re-synthesising a cluster that
            # is already in flight would duplicate the work and the email.
            existing = self.store.query_one(
                "SELECT id, state FROM pipelines WHERE cluster_id = ? "
                "ORDER BY created_at DESC LIMIT 1", (cluster.id,))
            if existing and existing["state"] not in (
                    State.REJECTED.value, State.COMPLETED.value, State.FAILED.value):
                continue
            self.machine.create(cluster_id=cluster.id, state=State.SYNTHESIZED,
                                data={"cluster_title": cluster.title})
            created += 1
        self.log.info(f"synthesis: {len(clusters)} clusters, {created} new pipelines",
                      status="ok")
        return {"clusters": len(clusters), "pipelines_created": created}

    async def job_verdict(self) -> dict[str, Any]:
        pending = (self.machine.in_state(State.SYNTHESIZED, limit=20)
                   + self.machine.in_state(State.VERDICT_PENDING, limit=20))
        decided = 0
        for pipe in pending:
            pipe = await self.driver.step(pipe)
            if pipe.state != State.SYNTHESIZED:
                decided += 1
        self.log.info(f"verdict pass: {decided}/{len(pending)} advanced", status="ok")
        return {"considered": len(pending), "advanced": decided}

    async def job_pipeline_tick(self) -> dict[str, Any]:
        """Advance every non-terminal pipeline by one step.

        One step per tick rather than running each to completion: it keeps any single
        pipeline from monopolising the key pool, and gives a crash a natural
        checkpoint between expensive stages.
        """
        active = self.machine.active(limit=20)
        advanced, failed = 0, 0
        for pipe in active:
            before = pipe.state
            pipe = await self.driver.step(pipe)
            if pipe.state == State.FAILED:
                failed += 1
            elif pipe.state != before:
                advanced += 1
        if active:
            self.log.info(
                f"pipeline tick: {advanced} advanced, {failed} failed, "
                f"{len(active)} active", status="ok")
        return {"active": len(active), "advanced": advanced, "failed": failed}

    async def job_monitor_sweep(self) -> dict[str, Any]:
        """Re-evaluate MONITOR research and retire what has gone nowhere.

        This is what makes the system track *maturity* rather than just novelty: a
        technique parked in March is reconsidered once implementations and
        replications appear, and one that never developed is eventually dropped so it
        stops costing attention.
        """
        cfg = self.cfg
        reeval_days = int(cfg.get("verdict.monitor_reevaluation_days", 7))
        expiry_days = int(cfg.get("verdict.monitor_expiry_days", 180))

        expired = self.store.query(
            "SELECT id, title FROM research_items WHERE lifecycle = 'MONITOR' "
            "AND julianday('now') - julianday(last_seen_at) > ?", (expiry_days,))
        for row in expired:
            self.store.execute(
                "UPDATE research_items SET lifecycle = 'DISMISSED', lifecycle_at = ?, "
                "dismiss_reason = ? WHERE id = ?",
                (utcnow(),
                 f"no new evidence in {expiry_days} days", row["id"]))

        ready = self.store.query(
            "SELECT p.id FROM pipelines p WHERE p.state = 'MONITORING' "
            "AND julianday('now') - julianday(p.updated_at) >= ?", (reeval_days,))
        requeued = 0
        for row in ready:
            pipe = self.machine.get(row["id"])
            self.machine.transition(
                pipe, State.VERDICT_PENDING,
                note="monitor sweep: re-evaluating against accumulated evidence")
            requeued += 1

        self.log.info(
            f"monitor sweep: {requeued} re-queued, {len(expired)} expired", status="ok")
        return {"requeued": requeued, "expired": len(expired)}

    async def job_global_qa(self) -> dict[str, Any]:
        """Whole-book QA, independent of any patch."""
        from ..agents import BookQAAgent
        report = await BookQAAgent().run(run_global=True, run_build=True,
                                         check_external_links=True)
        path = self.cfg.path("paths.reviews", "reviews") / (
            f"global-qa-{datetime.now(timezone.utc):%Y%m%d}.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_qa(report), encoding="utf-8")
        self.log.info(f"global QA: {report.summary} -> {path.name}", status="ok")
        return {"passed": report.passed, "summary": report.summary,
                "report": str(path.relative_to(self.cfg.root))}

    async def job_kb_reindex(self) -> dict[str, Any]:
        """Re-index if the manuscript changed. Cheap when it did not."""
        indexer = BookIndexer(self.store)
        fingerprint = indexer.manuscript_fingerprint()
        last = self.store.scalar(
            "SELECT value FROM schema_meta WHERE key = 'manuscript_fingerprint'")
        if last == fingerprint:
            self.log.debug("kb reindex: manuscript unchanged")
            return {"reindexed": False}

        book, structural = indexer.index_structure()
        semantic = await indexer.index_semantics(book)
        self.store.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES "
            "('manuscript_fingerprint', ?)", (fingerprint,))
        self.log.info(
            f"kb reindex: {structural.nodes_changed} changed, "
            f"{semantic.summaries} re-summarised", status="ok")
        return {"reindexed": True, "structural": structural.as_dict(),
                "semantic": semantic.as_dict()}

    # ── manual operations ─────────────────────────────────────────────────
    async def run_cycle(self, *, discover: bool = True, drive: bool = True) -> dict[str, Any]:
        """One full cycle: discover, synthesise, decide, then drive pipelines."""
        with run_scope() as run_id:
            self.store.start_run(run_id, trigger="cycle")
            summary: dict[str, Any] = {"run_id": run_id}
            try:
                if discover:
                    summary["discovery"] = await self.job_discovery()
                summary["synthesis"] = await self.job_synthesis()
                summary["verdict"] = await self.job_verdict()
                if drive:
                    summary["pipelines"] = await self._drive_all()
                self.store.end_run(run_id, "completed", summary)
            except Exception as exc:
                self.log.error(f"cycle failed: {type(exc).__name__}: {exc}")
                self.store.end_run(run_id, "failed", {"error": str(exc)})
                raise
            return summary

    async def _drive_all(self, *, max_pipelines: int = 10) -> dict[str, Any]:
        active = self.machine.active(limit=max_pipelines)
        outcomes: dict[str, str] = {}
        for pipe in active:
            final = await self.driver.run_to_completion(pipe)
            outcomes[final.id] = final.state.value
        return {"driven": len(active), "outcomes": outcomes}

    async def approve(self, pipeline_id: str, *, by: str = "human") -> Pipeline:
        pipe = self.machine.get(pipeline_id)
        if pipe.state != State.NEEDS_HUMAN:
            raise ValueError(
                f"{pipeline_id} is in {pipe.state.value}, not awaiting approval")
        if pipe.verdict_id:
            self.store.execute(
                "UPDATE verdicts SET approved_by = ?, approved_at = ? WHERE id = ?",
                (by, utcnow(), pipe.verdict_id))
        self.log.info(f"{pipeline_id} approved by {by}")
        return self.machine.transition(
            pipe, State.VERDICT_APPROVED, note=f"approved by {by}",
            data={"awaiting_approval": False, "approved_by": by})

    async def reject(self, pipeline_id: str, *, reason: str = "") -> Pipeline:
        pipe = self.machine.get(pipeline_id)
        return self.machine.transition(
            pipe, State.REJECTED, note=f"rejected by human: {reason}"[:400])

    async def resume_failed(self, *, limit: int = 20) -> int:
        """Return FAILED pipelines to a state they can be retried from."""
        rows = self.store.query(
            "SELECT id, previous_state FROM pipelines WHERE state = 'FAILED' LIMIT ?",
            (limit,))
        resumed = 0
        for row in rows:
            pipe = self.machine.get(row["id"])
            target = State(row["previous_state"]) if row["previous_state"] else State.DRAFTED
            if target in (State.FAILED, State.COMPLETED):
                target = State.DRAFTED
            try:
                self.machine.transition(pipe, target, note="resumed after failure")
                resumed += 1
            except Exception as exc:
                self.log.warn(f"could not resume {pipe.id}: {exc}")
        return resumed

    # ── reporting ─────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        counts = self.store.counts()
        lifecycle = {
            r["lifecycle"]: r["n"] for r in self.store.query(
                "SELECT lifecycle, COUNT(*) n FROM research_items GROUP BY lifecycle")
        }
        return {
            "pipelines": self.machine.summary(),
            "research_lifecycle": lifecycle,
            "counts": {k: v for k, v in counts.items() if v},
            "scheduler": self.scheduler.status(),
            "needs_human": [
                dict(r) for r in self.store.query(
                    "SELECT id, cluster_id, updated_at, last_error FROM pipelines "
                    "WHERE state = 'NEEDS_HUMAN' ORDER BY updated_at DESC LIMIT 10")
            ],
        }

    def _enabled_sources(self) -> set[str]:
        sources = self.cfg.get("research.sources", {}) or {}
        mapping = {
            "arxiv_agent": "arxiv",
            "scholarly_agent": "openalex",
            "github_research_agent": "github",
            "huggingface_agent": "huggingface",
            "research_blog_agent": "blogs",
            "conference_agent": "conferences",
            "benchmark_agent": "benchmarks",
            "community_agent": "hackernews",
        }
        return {
            agent for agent, key in mapping.items()
            if (sources.get(key) or {}).get("enabled", True)
        }

    async def run_forever(self) -> None:
        with run_scope() as run_id:
            self.store.start_run(run_id, trigger="daemon")
            self.log.info(f"living book daemon starting (run {run_id})")
            try:
                await self.scheduler.run_forever()
            finally:
                self.store.end_run(run_id, "completed")


def _render_qa(report: Any) -> str:
    lines = [f"# Global book QA — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
             f"**Result**: {'PASS' if report.passed else 'FAIL'}", "",
             f"`{report.summary}`", "", "## Deterministic checks", ""]
    for name, value in (report.deterministic or {}).items():
        if not isinstance(value, dict):
            continue
        lines.append(f"### {name} — {'ok' if value.get('ok', True) else 'FAIL'}")
        lines.append("")
        lines.append(f"```json\n{json.dumps(value, indent=2, default=str)[:4000]}\n```")
        lines.append("")
    for title, findings in (("Semantic findings", report.semantic),
                            ("Global findings", report.global_findings)):
        lines += [f"## {title}", ""]
        if not findings:
            lines += ["_none_", ""]
        for f in findings:
            lines.append(f"- **{f.severity}** `{f.kind}` — {f.detail}")
            if f.location:
                lines.append(f"  - at: {f.location[:200]}")
            if f.suggested_fix:
                lines.append(f"  - fix: {f.suggested_fix[:200]}")
        lines.append("")
    return "\n".join(lines)
