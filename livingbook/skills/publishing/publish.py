"""Git publishing and email reporting skills.

Both render from the same provenance report, so the changelog entry, the pull-request
body and the notification email tell one consistent story and cannot drift apart.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from ...config import get_config
from ...tools import AgentContext
from ...tools.git_tools import branch_name_for
from ..base import Skill, array_of, json_schema, string

CHANGELOG_SCHEMA = json_schema(
    {
        "title": string("One line, imperative mood, under 72 characters"),
        "summary": string("2-4 sentences for a reader of the changelog"),
        "why_it_matters": string("Why this research warranted changing the book"),
        "bullets": array_of(string(), "Specific changes made"),
    },
    ["title", "summary", "bullets"],
)


class GitPublishingSkill(Skill):
    name = "git_publishing"
    required_tools = ("git_status", "create_branch", "git_commit")
    optional_tools = ("git_push", "create_pull_request", "git_diff", "build_book",
                      "run_tests", "write_file", "gemini_generate",
                      "gemini_structured_output")

    async def run(
        self,
        ctx: AgentContext,
        *,
        provenance: dict[str, Any],
        changed_paths: list[str],
        topic: str,
        version: str,
        qa_summary: str = "",
        dry_run: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        cfg = get_config()
        result: dict[str, Any] = {"version": version, "dry_run": dry_run}

        status = await ctx.call("git_status")
        if not status.get("is_repo"):
            return {**result, "published": False,
                    "reason": "not a git repository; run `livingbook git init` first"}

        branch = branch_name_for(topic)
        await ctx.call("create_branch", name=branch)
        result["branch"] = branch

        narrative = await self._narrative(ctx, provenance, changed_paths)
        changelog_path = f"changelog/{version}.md"
        changelog = self._render_changelog(version, narrative, provenance,
                                           changed_paths, qa_summary)
        if "write_file" in ctx.allowed_tools:
            await ctx.call("write_file", path=changelog_path, content=changelog)
        result["changelog_path"] = changelog_path

        # Build on the branch before committing: a change that does not compile must
        # never reach the remote, whatever the earlier QA said.
        if "build_book" in ctx.allowed_tools and cfg.get("qa.run_build", True):
            build = await ctx.try_call("build_book", default={"ok": False})
            result["build"] = {"ok": build.get("ok"), "pages": build.get("pages"),
                               "errors": build.get("errors", [])[:5]}
            if not build.get("ok"):
                return {**result, "published": False,
                        "reason": "build failed on the agent branch; not committing"}

        verdict = (provenance.get("verdict") or {})
        cluster = (provenance.get("cluster") or {})
        trailers = {
            "Living-Book-Version": version,
            "Research-Cluster": cluster.get("id", ""),
            "Verdict": verdict.get("decision", ""),
            "Maturity": cluster.get("maturity", ""),
            "Pipeline": provenance.get("pipeline_id", ""),
            "Sources": ", ".join(
                f"{s['source']}:{(s.get('url') or '')[:60]}"
                for s in (provenance.get("sources") or [])[:4]),
        }
        message = self._commit_message(narrative, verdict, changed_paths)

        commit = await ctx.call(
            "git_commit", message=message,
            paths=changed_paths + [changelog_path],
            trailers={k: v for k, v in trailers.items() if v})
        result["commit"] = commit
        if not commit.get("committed"):
            return {**result, "published": False,
                    "reason": commit.get("reason", "nothing to commit")}

        if dry_run or not cfg.get("git.push_enabled", True):
            return {**result, "published": False, "reason": "push disabled (dry run)",
                    "committed_locally": True}

        push = await ctx.try_call("git_push", default=None, branch=branch)
        result["push"] = push
        if not push or not push.get("pushed"):
            return {**result, "published": False,
                    "reason": (push or {}).get("reason", "push failed"),
                    "committed_locally": True}

        if cfg.get("git.create_pull_request", True) and "create_pull_request" in ctx.allowed_tools:
            body = self._pr_body(narrative, provenance, changed_paths, qa_summary, version)
            pr = await ctx.try_call(
                "create_pull_request", default=None,
                title=narrative.get("title", f"Living Book update {version}"),
                body=body, head=branch)
            result["pull_request"] = pr

        result["published"] = True
        return result

    # -- narrative ---------------------------------------------------------
    async def _narrative(
        self, ctx: AgentContext, provenance: dict[str, Any], changed_paths: list[str],
    ) -> dict[str, Any]:
        cluster = provenance.get("cluster") or {}
        verdict = provenance.get("verdict") or {}
        sources = provenance.get("sources") or []

        prompt = (
            "Write the changelog entry for an automated update to a Vietnamese NLP "
            "textbook. Write in English, for a maintainer reviewing what the system did.\n\n"
            "Be concrete and factual. Do not oversell — this is a record, not an "
            "announcement.\n\n"
            f"RESEARCH: {cluster.get('title','')}\n"
            f"MATURITY: {cluster.get('maturity','')}\n"
            f"CONCEPTS: {', '.join(cluster.get('concepts', [])[:8])}\n"
            f"DECISION: {verdict.get('decision','')}\n"
            f"RATIONALE: {verdict.get('rationale','')[:2000]}\n"
            f"FILES CHANGED: {', '.join(changed_paths)}\n"
            f"SOURCES:\n" + "\n".join(
                f"  - [{s.get('source')}] {s.get('title','')[:100]}"
                for s in sources[:8])
        )
        result = await ctx.try_call("gemini_structured_output", default=None,
                                    prompt=prompt, schema=CHANGELOG_SCHEMA,
                                    temperature=0.3, role="fast")
        if result:
            return result["data"]
        return {
            "title": f"Update from research: {cluster.get('title','')[:60]}",
            "summary": verdict.get("rationale", "")[:400],
            "bullets": [f"Updated {p}" for p in changed_paths],
        }

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def _commit_message(narrative: dict[str, Any], verdict: dict[str, Any],
                        changed_paths: list[str]) -> str:
        decision = (verdict.get("decision") or "update").lower()
        kind = {
            "add_reference": "docs", "add_footnote": "docs",
            "extend_section": "feat", "add_new_section": "feat",
            "rewrite_section": "refactor", "replace_obsolete_content": "fix",
        }.get(decision, "docs")
        title = narrative.get("title", "update manuscript")[:68]
        body = narrative.get("summary", "")
        bullets = "\n".join(f"- {b}" for b in (narrative.get("bullets") or [])[:8])
        return f"{kind}(book): {title}\n\n{body}\n\n{bullets}".strip()

    @staticmethod
    def _render_changelog(
        version: str, narrative: dict[str, Any], provenance: dict[str, Any],
        changed_paths: list[str], qa_summary: str,
    ) -> str:
        cluster = provenance.get("cluster") or {}
        verdict = provenance.get("verdict") or {}
        sources = provenance.get("sources") or []
        evidence = provenance.get("evidence") or []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# {version} — {narrative.get('title','Manuscript update')}",
            "",
            f"*{now}*",
            "",
            narrative.get("summary", ""),
            "",
            "## Why this changed the book",
            "",
            narrative.get("why_it_matters") or verdict.get("rationale", ""),
            "",
            "## Changes",
            "",
        ]
        lines += [f"- {b}" for b in (narrative.get("bullets") or [])]
        lines += ["", "## Files", ""]
        lines += [f"- `{p}`" for p in changed_paths]

        lines += ["", "## Research provenance", "",
                  f"- **Cluster**: {cluster.get('title','')} (`{cluster.get('id','')}`)",
                  f"- **Maturity**: {cluster.get('maturity','')}",
                  f"- **Decision**: {verdict.get('decision','')} "
                  f"(scope: {verdict.get('scope','')})",
                  f"- **Concepts**: {', '.join(cluster.get('concepts', [])[:10])}",
                  "", "### Sources", ""]
        for s in sources[:15]:
            lines.append(f"- [{s.get('source')}] [{s.get('title','')}]({s.get('url','')})"
                         + (f" — {s.get('published_at')}" if s.get("published_at") else ""))

        if evidence:
            lines += ["", "### Evidence", ""]
            for e in evidence[:15]:
                lines.append(f"- `{e.get('kind')}`/`{e.get('strength')}`: "
                             f"{e.get('statement','')[:220]}")

        if qa_summary:
            lines += ["", "## Verification", "", f"```\n{qa_summary}\n```"]

        lines += ["", "---", "",
                  "*Generated by the Living Book autonomous pipeline. "
                  "Every statement above traces to a recorded artifact; see "
                  "`state/artifacts/` for the full chain.*", ""]
        return "\n".join(lines)

    @staticmethod
    def _pr_body(narrative: dict[str, Any], provenance: dict[str, Any],
                 changed_paths: list[str], qa_summary: str, version: str) -> str:
        cluster = provenance.get("cluster") or {}
        verdict = provenance.get("verdict") or {}
        sources = provenance.get("sources") or []
        transitions = provenance.get("transitions") or []

        lines = [
            narrative.get("summary", ""), "",
            "## Why", "", verdict.get("rationale", "")[:3000], "",
            "## Changes", "",
        ]
        lines += [f"- {b}" for b in (narrative.get("bullets") or [])]
        lines += ["", "## Files changed", ""]
        lines += [f"- `{p}`" for p in changed_paths]
        lines += ["", "## Research", "",
                  f"- **Cluster**: {cluster.get('title','')}",
                  f"- **Maturity**: `{cluster.get('maturity','')}`",
                  f"- **Decision**: `{verdict.get('decision','')}`",
                  "", "<details><summary>Sources</summary>", ""]
        for s in sources[:20]:
            lines.append(f"- [{s.get('source')}] [{s.get('title','')}]({s.get('url','')})")
        lines += ["", "</details>", "", "## Verification", "",
                  f"```\n{qa_summary or 'n/a'}\n```", "",
                  "<details><summary>Pipeline trace</summary>", ""]
        for t in transitions[-20:]:
            lines.append(f"- `{t.get('from')}` -> `{t.get('to')}` "
                         f"({t.get('at','')}){' — ' + t['note'] if t.get('note') else ''}")
        lines += ["", "</details>", "",
                  f"Version `{version}` · pipeline `{provenance.get('pipeline_id','')}`",
                  "", "🤖 Generated by the Living Book autonomous pipeline."]
        return "\n".join(lines)


class EmailReportingSkill(Skill):
    """Compose and send the update notification.

    Only ever invoked from the EMAIL pipeline state, which is reachable only through
    APPROVED. MONITOR and IGNORE verdicts terminate before it, so research that did not
    change the book cannot produce an update email.
    """

    name = "email_reporting"
    required_tools = ("send_email",)
    optional_tools = ("kb_query", "gemini_generate", "gemini_structured_output")

    async def run(
        self,
        ctx: AgentContext,
        *,
        provenance: dict[str, Any],
        git_result: dict[str, Any],
        version: str,
        changed_paths: list[str],
        qa_summary: str = "",
        citation_changes: list[dict[str, Any]] | None = None,
        figure_changes: list[dict[str, Any]] | None = None,
        verification_status: dict[str, str] | None = None,
        pipeline_id: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        cluster = provenance.get("cluster") or {}
        verdict = provenance.get("verdict") or {}

        subject = (f"[Living Book {version}] "
                   f"{cluster.get('title', 'Manuscript updated')[:70]}")
        html_body = self._render_html(
            version=version, provenance=provenance, git_result=git_result,
            changed_paths=changed_paths, qa_summary=qa_summary,
            citation_changes=citation_changes or [],
            figure_changes=figure_changes or [],
            verification_status=verification_status or {},
        )
        return await ctx.call(
            "send_email", subject=subject, html=html_body,
            pipeline_id=pipeline_id,
            # One notification per pipeline version, so a retried delivery step
            # cannot produce a duplicate email.
            dedupe_key=f"livingbook:{pipeline_id}:{version}",
        )

    def _render_html(
        self, *, version: str, provenance: dict[str, Any], git_result: dict[str, Any],
        changed_paths: list[str], qa_summary: str,
        citation_changes: list[dict[str, Any]], figure_changes: list[dict[str, Any]],
        verification_status: dict[str, str],
    ) -> str:
        cluster = provenance.get("cluster") or {}
        verdict = provenance.get("verdict") or {}
        sources = provenance.get("sources") or []
        evidence = provenance.get("evidence") or []
        e = html.escape

        def section(title: str, inner: str) -> str:
            return (
                f'<tr><td style="padding:0 40px 8px;">'
                f'<p style="margin:22px 0 8px;font-size:12px;font-weight:800;'
                f'color:#7c3aed;text-transform:uppercase;letter-spacing:0.7px;">{title}</p>'
                f'{inner}</td></tr>'
            )

        def para(text: str) -> str:
            return (f'<p style="margin:0 0 10px;font-size:14px;color:#334155;'
                    f'line-height:1.65;">{text}</p>')

        def items(rows: list[str]) -> str:
            if not rows:
                return para('<em style="color:#94a3b8;">none</em>')
            lis = "".join(
                f'<li style="margin:0 0 6px;font-size:13px;color:#334155;'
                f'line-height:1.55;">{r}</li>' for r in rows)
            return f'<ul style="margin:0 0 10px;padding-left:20px;">{lis}</ul>'

        pr = (git_result.get("pull_request") or {})
        pr_url = pr.get("url", "")
        commit = (git_result.get("commit") or {})

        status_rows = [
            f'<b>{e(k.replace("_", " "))}</b>: '
            f'<span style="color:{"#16a34a" if v in ("pass", "ok", "passed") else "#dc2626"};">'
            f'{e(str(v))}</span>'
            for k, v in verification_status.items()
        ]

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/></head>
<body style="margin:0;padding:0;background:#f8f7ff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f8f7ff;padding:36px 14px;">
<tr><td align="center">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:640px;background:#ffffff;border-radius:22px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">

<tr><td style="background:linear-gradient(135deg,#1b145a 0%,#4c3ba5 100%);padding:30px 40px 26px;">
  <h1 style="margin:0;color:#fff;font-size:20px;font-weight:800;letter-spacing:-0.3px;">Living Book — cập nhật bản thảo</h1>
  <p style="margin:6px 0 0;color:rgba(255,255,255,0.68);font-size:13px;">
    {e(version)} · {e(str(verdict.get('decision','')))} · maturity {e(str(cluster.get('maturity','')))}
  </p>
</td></tr>

{section("Research discovered", para(e(cluster.get('title',''))) + para(e((cluster.get('summary') or provenance.get('summary') or '')[:600])))}
{section("Why it matters", para(e(verdict.get('rationale','')[:1400])))}
{section("What changed", items([e(b) for b in changed_paths]))}
{section("Citation changes", items([
    e(f"{c.get('action','added')} {c.get('bib_key','')}: {c.get('title','')[:90]}")
    for c in citation_changes]))}
{section("Figure changes", items([
    e(f"{f.get('action','added')} {f.get('key','')} ({f.get('license','?')})"
      + (f" — {f.get('attribution','')}" if f.get('attribution') else ""))
    for f in figure_changes]))}
{section("Verification", items(status_rows) + (f'<pre style="margin:6px 0 0;padding:10px;background:#f8fafc;border-radius:8px;font-size:11px;color:#475569;white-space:pre-wrap;">{e(qa_summary)}</pre>' if qa_summary else ""))}
{section("Evidence", items([
    e(f"[{ev.get('kind')}/{ev.get('strength')}] {ev.get('statement','')[:160]}")
    for ev in evidence[:8]]))}
{section("Sources", items([
    f'<a href="{e(s.get("url",""))}" style="color:#4f46e5;text-decoration:none;">'
    f'[{e(str(s.get("source")))}] {e(s.get("title","")[:95])}</a>'
    for s in sources[:12]]))}

<tr><td style="padding:14px 40px 30px;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f8f7ff;border-radius:14px;">
  <tr><td style="padding:16px 18px;">
    <p style="margin:0 0 6px;font-size:12px;color:#64748b;">
      <b>Branch:</b> <code>{e(str(git_result.get('branch','n/a')))}</code></p>
    <p style="margin:0 0 6px;font-size:12px;color:#64748b;">
      <b>Commit:</b> <code>{e(str(commit.get('short_sha','n/a')))}</code></p>
    <p style="margin:0;font-size:12px;color:#64748b;">
      <b>Pipeline:</b> <code>{e(str(provenance.get('pipeline_id','')))}</code></p>
  </td></tr></table>
  {"" if not pr_url else f'''
  <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:16px;"><tr><td align="center">
    <a href="{e(pr_url)}" style="display:inline-block;background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;text-decoration:none;font-size:14px;font-weight:700;padding:13px 32px;border-radius:12px;">Review the pull request →</a>
  </td></tr></table>'''}
</td></tr>

<tr><td style="padding:16px 40px 28px;border-top:1px solid #f1f5f9;">
  <p style="margin:0;font-size:11px;color:#94a3b8;text-align:center;line-height:1.6;">
    Sent because the manuscript actually changed. Research that is only being monitored
    does not trigger this email.<br/>
    Every claim above traces to a recorded artifact.
  </p>
</td></tr>

</table></td></tr></table></body></html>"""
