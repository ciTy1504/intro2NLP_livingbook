"""Delivery agents: Git and Email.

The Git agent escalates to a human rather than retrying on failure — a half-published
change is worse than an unpublished one, and the causes (auth, a dirty tree, a
rejected push) are not things a retry fixes.

The Email agent is only ever reachable through the EMAIL pipeline state, which is only
reachable through APPROVED. That is the structural guarantee behind "no manuscript
change, no update email".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ...config import get_config
from ...state.artifacts import get_artifacts
from ...state.store import utcnow
from ..base import BaseAgent


class GitAgent(BaseAgent[dict[str, Any]]):
    name = "git_agent"
    uses_skills = ("git_publishing",)

    async def execute(
        self, *, pipeline_id: str, changed_paths: list[str], topic: str,
        qa_summary: str = "", version: str | None = None, dry_run: bool = False, **_: Any,
    ) -> dict[str, Any]:
        provenance = get_artifacts().provenance_report(pipeline_id)
        version = version or self._next_version()

        result = await self.skill("git_publishing")(
            self.ctx, provenance=provenance, changed_paths=changed_paths,
            topic=topic, version=version, qa_summary=qa_summary, dry_run=dry_run)

        if result.get("published") or result.get("committed_locally"):
            commit = result.get("commit") or {}
            pr = result.get("pull_request") or {}
            self.store.execute(
                "INSERT OR REPLACE INTO versions (version, created_at, commit_sha, "
                "branch, pull_request_url, changelog_path, pipeline_ids_json, summary) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (version, utcnow(), commit.get("sha"), result.get("branch"),
                 pr.get("url"), result.get("changelog_path"),
                 f'["{pipeline_id}"]',
                 (provenance.get("verdict") or {}).get("rationale", "")[:800]),
            )
        self.save("git_result", result, meta={"version": version,
                                              "published": result.get("published")})
        return {**result, "version": version, "provenance": provenance}

    def _next_version(self) -> str:
        """Date-based with a same-day counter: v2026.09.16.1, .2, …

        Semantic versioning does not describe a book. A reader wants to know when a
        change landed and in what order.
        """
        today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        n = int(self.store.scalar(
            "SELECT COUNT(*) FROM versions WHERE version LIKE ?", (f"v{today}%",)) or 0)
        return f"v{today}.{n + 1}"


class EmailAgent(BaseAgent[dict[str, Any]]):
    name = "email_agent"
    uses_skills = ("email_reporting",)

    async def execute(
        self, *, pipeline_id: str, git_result: dict[str, Any], version: str,
        changed_paths: list[str], qa_summary: str = "",
        citation_changes: list[dict[str, Any]] | None = None,
        figure_changes: list[dict[str, Any]] | None = None,
        verification_status: dict[str, str] | None = None,
        manuscript_changed: bool = True, **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()

        # Belt and braces. The state machine already prevents this, but an email
        # announcing a change that did not happen is the one failure that would cost
        # the user's trust in every other email.
        if cfg.get("email.send_only_on_manuscript_change", True) and not manuscript_changed:
            self.log.info("no manuscript change; suppressing update email")
            return {"sent": False, "reason": "no manuscript change"}

        provenance = get_artifacts().provenance_report(pipeline_id)
        result = await self.skill("email_reporting")(
            self.ctx, provenance=provenance, git_result=git_result, version=version,
            changed_paths=changed_paths, qa_summary=qa_summary,
            citation_changes=citation_changes, figure_changes=figure_changes,
            verification_status=verification_status, pipeline_id=pipeline_id)

        self.save("email", {"result": result, "version": version},
                  meta={"sent": result.get("sent")})
        return result
