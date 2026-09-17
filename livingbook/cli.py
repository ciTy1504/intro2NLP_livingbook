"""Living Book command line.

    python -m livingbook.cli doctor          check the environment and every contract
    python -m livingbook.cli index           build the book knowledge base
    python -m livingbook.cli discover        run one discovery pass
    python -m livingbook.cli cycle           discover -> synthesise -> decide -> drive
    python -m livingbook.cli daemon          run continuously on the schedule
    python -m livingbook.cli status          what the system is doing
    python -m livingbook.cli approve <id>    approve a gated change
    python -m livingbook.cli qa              run full-book QA now
    python -m livingbook.cli llm probe       re-measure model availability
    python -m livingbook.cli git init        attach the Living Book remote
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook import __version__                                   # noqa: E402
from livingbook.config import get_config, get_secrets                # noqa: E402
from livingbook.obs import get_logger, run_scope                     # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────


def _print(data: Any) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(data)


def _h(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("─" * min(len(title), 78))


# ── commands ──────────────────────────────────────────────────────────────


async def cmd_doctor(args: argparse.Namespace) -> int:
    """Check everything a run depends on, before an unattended run depends on it."""
    from livingbook.agents import AGENTS, validate_all
    from livingbook.skills import SKILLS
    from livingbook.state.store import get_store
    from livingbook.tools import REGISTRY

    cfg = get_config()
    secrets = get_secrets()
    problems: list[str] = []
    warnings: list[str] = []

    _h("Configuration")
    print(f"  repo root          {cfg.root}")
    print(f"  manuscript         {cfg.manuscript_dir}")
    print(f"  remote             {cfg.get('git.remote_url')}")
    print(f"  version            {__version__}")

    _h("Registries")
    print(f"  tools              {len(REGISTRY.names())}")
    print(f"  skills             {len(SKILLS)}")
    print(f"  agents             {len(AGENTS)}")
    contract_problems = validate_all()
    if contract_problems:
        problems.extend(contract_problems)
        print(f"  contracts          {len(contract_problems)} PROBLEMS")
    else:
        print("  contracts          all valid")

    _h("Manuscript")
    if not cfg.main_tex.exists():
        problems.append(f"main tex missing: {cfg.main_tex}")
    else:
        from livingbook.knowledge.latex import LatexParser
        book = LatexParser(cfg.manuscript_dir,
                           cfg.get("project.main_tex", "main.tex")).parse()
        stats = book.stats()
        print(f"  files              {stats['files']}")
        print(f"  chapters/sections  {stats['chapter']}/{stats['section']}")
        print(f"  words              {stats['words']:,}")
        print(f"  figures            {stats['figures']}")
        orphans = {c for n in book.nodes for c in n.cites} - book.bib_keys
        if orphans:
            warnings.append(f"{len(orphans)} orphan citations: {sorted(orphans)[:5]}")
        print(f"  orphan citations   {len(orphans)}")

    _h("Knowledge base")
    store = get_store()
    counts = store.counts()
    for key in ("book_nodes", "node_summaries", "concepts", "claims",
                "bib_entries", "graph_edges", "embeddings", "research_items",
                "clusters", "pipelines"):
        print(f"  {key:18s} {counts.get(key, 0)}")
    if counts.get("node_summaries", 0) == 0:
        warnings.append("knowledge base not indexed — run `livingbook index`")

    _h("Secrets")
    for key, required, why in (
        ("GEMINI_KEYPOOL_PATH", False, "key pool file location"),
        ("GITHUB_TOKEN", False, "required to push and open pull requests"),
        ("GMAIL_USER", False, "required to send notification email"),
        ("GMAIL_PASS", False, "Google App Password"),
        ("EMAIL_RECIPIENTS", False, "who receives update notifications"),
        ("SEMANTIC_SCHOLAR_API_KEY", False, "optional; enables Semantic Scholar"),
    ):
        present = secrets.has(key)
        mark = "set" if present else "not set"
        print(f"  {key:26s} {mark:8s} {'' if present else '— ' + why}")
        if not present and key in ("GITHUB_TOKEN", "GMAIL_USER", "GMAIL_PASS"):
            warnings.append(f"{key} not set: {why}")

    _h("LLM provider")
    try:
        from livingbook.llm import get_provider
        provider = get_provider()
        status = provider.status()
        pool = status["keypool"]
        print(f"  keys               {pool['total']} "
              f"({pool['available']} available, {pool['disabled']} disabled)")
        for role, chain in status["roles"].items():
            print(f"  role {role:10s}    {' -> '.join(chain)}")
        if pool["disabled"] > pool["total"] * 0.25:
            warnings.append(
                f"{pool['disabled']}/{pool['total']} keys disabled; "
                "consider `livingbook llm revive`")
        await provider.aclose()
    except Exception as exc:
        problems.append(f"LLM provider unavailable: {type(exc).__name__}: {exc}")

    _h("Toolchain")
    from livingbook.tools.book import _find_binary
    for binary, why in (("xelatex", "required to build the book"),
                        ("bibtex", "required to resolve citations"),
                        ("git", "required to publish")):
        found = _find_binary(binary)
        print(f"  {binary:18s} {'found' if found else 'MISSING'}")
        if not found:
            (problems if binary == "git" else warnings).append(f"{binary} not found — {why}")

    _h("Result")
    for w in warnings:
        print(f"  \033[33mwarning\033[0m  {w}")
    for p in problems:
        print(f"  \033[31mproblem\033[0m  {p}")
    if not problems and not warnings:
        print("  \033[32mall checks passed\033[0m")
    elif not problems:
        print(f"  \033[32musable\033[0m — {len(warnings)} warning(s)")
    else:
        print(f"  \033[31m{len(problems)} problem(s)\033[0m must be fixed")
    print()
    return 1 if problems else 0


async def cmd_index(args: argparse.Namespace) -> int:
    from livingbook.knowledge.graph import KnowledgeGraph
    from livingbook.knowledge.indexer import BookIndexer
    from livingbook.llm import get_provider

    log = get_logger()
    with run_scope():
        indexer = BookIndexer()
        book, structural = indexer.index_structure()
        _print(structural.as_dict())
        if not args.structure_only:
            semantic = await indexer.index_semantics(
                book, force=args.force, concurrency=args.concurrency, limit=args.limit)
            _print(semantic.as_dict())
        _print(KnowledgeGraph().stats())
        await get_provider().aclose()
    return 0


async def cmd_discover(args: argparse.Namespace) -> int:
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    with run_scope():
        result = await orch.job_discovery()
    _print(result)
    await get_provider().aclose()
    return 0


async def cmd_synthesize(args: argparse.Namespace) -> int:
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    with run_scope():
        result = await orch.job_synthesis()
    _print(result)
    await get_provider().aclose()
    return 0


async def cmd_cycle(args: argparse.Namespace) -> int:
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    result = await orch.run_cycle(discover=not args.no_discover, drive=not args.no_drive)
    _print(result)
    provider = get_provider()
    _h("LLM usage")
    _print(provider.status()["usage"])
    await provider.aclose()
    return 0


async def cmd_tick(args: argparse.Namespace) -> int:
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    with run_scope():
        result = await orch.job_pipeline_tick()
    _print(result)
    await get_provider().aclose()
    return 0


async def cmd_daemon(args: argparse.Namespace) -> int:
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    if args.reset_schedule:
        orch.scheduler.reset()
    try:
        await orch.run_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        get_logger().info("daemon stopped")
    return 0


async def cmd_status(args: argparse.Namespace) -> int:
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    status = orch.status()

    _h("Pipelines")
    if status["pipelines"]:
        for state, n in status["pipelines"].items():
            print(f"  {state:20s} {n}")
    else:
        print("  none yet")

    _h("Research memory")
    for lifecycle, n in (status["research_lifecycle"] or {}).items():
        print(f"  {lifecycle:20s} {n}")

    _h("Knowledge base")
    for key, n in status["counts"].items():
        print(f"  {key:20s} {n}")

    _h("Schedule")
    for job in status["scheduler"]:
        print(f"  {job['name']:16s} last={job['last_run_at'] or '-':25s} "
              f"next={job['next_run_at'] or 'now':25s} "
              f"status={job['last_status'] or '-'}")

    if status["needs_human"]:
        _h("Awaiting your decision")
        for row in status["needs_human"]:
            print(f"  {row['id']}  ({row['updated_at']})")
            print(f"    review: reviews/{row['id']}.md")
            print(f"    approve: python -m livingbook.cli approve {row['id']}")
    print()
    return 0


async def cmd_approve(args: argparse.Namespace) -> int:
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    pipe = await orch.approve(args.pipeline_id, by=args.by)
    print(f"{pipe.id} -> {pipe.state.value}")
    if args.run:
        final = await orch.driver.run_to_completion(pipe)
        print(f"{final.id} -> {final.state.value}")
    await get_provider().aclose()
    return 0


async def cmd_reject(args: argparse.Namespace) -> int:
    from livingbook.orchestrator import Orchestrator

    orch = Orchestrator()
    pipe = await orch.reject(args.pipeline_id, reason=args.reason)
    print(f"{pipe.id} -> {pipe.state.value}")
    return 0


async def cmd_qa(args: argparse.Namespace) -> int:
    from livingbook.agents import BookQAAgent
    from livingbook.llm import get_provider

    with run_scope():
        report = await BookQAAgent().run(
            run_global=not args.no_global, run_build=not args.no_build,
            check_external_links=args.links)
    _h("Deterministic")
    for name, value in (report.deterministic or {}).items():
        if isinstance(value, dict):
            print(f"  {name:14s} {'ok' if value.get('ok', True) else 'FAIL'}")
            for key in ("errors", "orphan_citations", "missing", "broken", "problems"):
                if value.get(key):
                    print(f"    {key}: {json.dumps(value[key], default=str)[:400]}")
    for title, findings in (("Semantic", report.semantic),
                            ("Global", report.global_findings)):
        if findings:
            _h(title)
            for f in findings:
                print(f"  [{f.severity}] {f.kind}: {f.detail[:200]}")
    _h("Result")
    print(f"  {'PASS' if report.passed else 'FAIL'} — {report.summary}")
    await get_provider().aclose()
    return 0 if report.passed else 1


async def cmd_llm(args: argparse.Namespace) -> int:
    from livingbook.llm import GenerationOptions, get_provider

    provider = get_provider()

    if args.llm_command == "status":
        _print(provider.status())

    elif args.llm_command == "probe":
        _h("Model availability")
        for role in provider.selector.roles():
            for model in provider.selector.chain(role):
                try:
                    body = {"contents": [{"parts": [{"text": "Reply: OK"}]}],
                            "generationConfig": {"maxOutputTokens": 2000,
                                                 "temperature": 0}}
                    if "embedding" in model:
                        await provider.embed(["probe"], role=role)
                        print(f"  {role:10s} {model:28s} ok (embedding)")
                        break
                    await provider._request(model, "generateContent", body, timeout=45)
                    print(f"  {role:10s} {model:28s} \033[32mok\033[0m")
                except Exception as exc:
                    print(f"  {role:10s} {model:28s} \033[31m{type(exc).__name__}\033[0m: "
                          f"{str(exc)[:70]}")
        _h("Key pool")
        _print(provider.pool.status())

    elif args.llm_command == "revive":
        n = await provider.pool.revive_disabled()
        print(f"revived {n} disabled keys")

    elif args.llm_command == "models":
        for m in await provider.list_models():
            print(f"  {m}")

    elif args.llm_command == "keys":
        rows = provider.pool.per_key_status()
        _h(f"Key pool ({len(rows)} keys)")
        for r in rows[:args.limit]:
            print(f"  {r['key']}  {r['state']:8s} req={r['requests']:5d} "
                  f"ok={r['successes']:5d} fail={r['failures']:5d} "
                  f"{r['reason'] or ''}")

    await provider.aclose()
    return 0


async def cmd_kb(args: argparse.Namespace) -> int:
    from livingbook.knowledge.graph import KnowledgeGraph
    from livingbook.knowledge.retrieval import BookRetriever
    from livingbook.llm import get_provider

    retriever = BookRetriever()
    graph = KnowledgeGraph()

    if args.kb_command == "outline":
        print(retriever.outline(max_depth=args.depth,
                                with_summaries=not args.no_summaries))
    elif args.kb_command == "search":
        hits = await retriever.retrieve(args.query, limit=args.limit)
        for h in hits:
            print(f"\n  [{h.score:5.2f}] {h.ref}")
            print(f"        {h.file}:{h.start_line}-{h.end_line}  via {h.matched_by}")
            if h.summary:
                print(f"        {h.summary[:220]}")
    elif args.kb_command == "impact":
        for hit in graph.impact_of_concepts(args.concepts, limit=args.limit):
            print(f"\n  [{hit.score:5.2f}] {hit.number} {hit.title}")
            print(f"        {hit.file}:{hit.start_line}  {hit.reasons}")
            print(f"        concepts: {', '.join(hit.concepts[:8])}")
            print(f"        citations: {', '.join(hit.citations[:8])}")
    elif args.kb_command == "concept":
        _print(graph.concept_coverage(args.name))
    elif args.kb_command == "citation":
        _print(graph.citation_neighbourhood(args.key))
    elif args.kb_command == "stats":
        _print(graph.stats())
    elif args.kb_command == "duplication":
        _print(graph.duplicate_concept_coverage(min_nodes=args.min_nodes))

    await get_provider().aclose()
    return 0


async def cmd_git(args: argparse.Namespace) -> int:
    from livingbook.tools import system_context

    ctx = system_context("cli")
    if args.git_command == "init":
        result = await ctx.call("init_repository")
        _print(result)
    elif args.git_command == "status":
        _print(await ctx.call("git_status"))
    elif args.git_command == "log":
        for entry in await ctx.call("git_log", limit=args.limit):
            print(f"  {entry['short_sha']}  {entry['date'][:10]}  {entry['subject'][:70]}")
    return 0


async def cmd_trace(args: argparse.Namespace) -> int:
    from livingbook.state.artifacts import get_artifacts
    from livingbook.state.store import get_store

    if args.pipeline_id:
        report = get_artifacts().provenance_report(args.pipeline_id)
        _print(report)
        return 0

    store = get_store()
    rows = store.trace(run_id=args.run_id, research_id=args.research_id, limit=args.limit)
    for r in rows:
        who = r["agent"] or "system"
        if r["skill"]:
            who += f"/{r['skill']}"
        if r["tool"]:
            who += f":{r['tool']}"
        print(f"  {r['ts'][11:19]}  {who:42s} {r['state'] or '-':18s} {r['message'][:80]}")
    return 0


async def cmd_visual(args: argparse.Namespace) -> int:
    """Resolve outstanding \\bookimage placeholders."""
    from livingbook.agents import VisualNeedDetector
    from livingbook.llm import get_provider
    from livingbook.orchestrator.visual_flow import run_visual_pipeline

    with run_scope():
        detector = VisualNeedDetector()
        requirements = await detector.run(include_unresolved_placeholders=True)
        if args.key:
            requirements = [r for r in requirements if r.key == args.key]
        if args.limit:
            requirements = requirements[:args.limit]
        if not requirements:
            print("no unresolved figure requirements")
            return 0
        print(f"resolving {len(requirements)} figure requirement(s)")
        placed = await run_visual_pipeline(requirements)
    _print(placed)
    await get_provider().aclose()
    return 0


async def cmd_email(args: argparse.Namespace) -> int:
    from livingbook.tools import system_context

    ctx = system_context("cli")
    if args.email_command == "status":
        _print(await ctx.call("email_status"))
    elif args.email_command == "test":
        result = await ctx.call(
            "send_email",
            subject="[Living Book] test message",
            html="<p>If you are reading this, email delivery works.</p>"
                 "<p>Sent by the Living Book pipeline.</p>",
            text="If you are reading this, email delivery works.")
        _print(result)
    return 0



async def cmd_mcp(args: argparse.Namespace) -> int:
    """MCP: serve the Living Book, or inspect external servers it can mount."""
    if args.mcp_command == "serve":
        from livingbook.mcp.server import main as serve
        serve()
        return 0

    if args.mcp_command == "tools":
        from livingbook.mcp.server import mcp as server
        listed = await server.list_tools()
        _h(f"Exposed over MCP ({len(listed)} tools)")
        for t in listed:
            print(f"  {t.name:22s} {(t.description or '').splitlines()[0][:72]}")
        return 0

    if args.mcp_command == "external":
        from livingbook.mcp import probe_servers
        _print(await probe_servers())
        return 0

    return 0



async def cmd_stuck(args: argparse.Namespace) -> int:
    """Show pipelines that need an operator, and optionally retry them."""
    from livingbook.llm import get_provider
    from livingbook.orchestrator import Orchestrator
    from livingbook.state.machine import State

    orch = Orchestrator()

    if args.retry:
        pipe = await orch.retry(
            args.retry,
            from_state=State(args.from_state) if args.from_state else None,
            reset_revisions=args.reset_revisions,
            note=args.note or "retried by operator")
        print(f"{pipe.id} -> {pipe.state.value}")
        if args.run:
            final = await orch.driver.run_to_completion(pipe)
            print(f"{final.id} -> {final.state.value}")
        await get_provider().aclose()
        return 0

    if args.resume_all:
        n = await orch.resume_failed()
        print(f"resumed {n} failed pipeline(s)")
        return 0

    rows = orch.stuck()
    if not rows:
        print("nothing is stuck")
        return 0
    _h(f"Needs an operator ({len(rows)})")
    for r in rows:
        print(f"\n  {r['id']}  [{r['state']}]  revisions={r['revisions']}  {r['updated_at']}")
        if r.get("title"):
            print(f"    research: [{r['maturity']}] {r['title'][:70]}")
        if r.get("last_error"):
            print(f"    error   : {r['last_error'][:200]}")
        print(f"    review  : reviews/{r['id']}.md")
        print(f"    retry   : python -m livingbook.cli stuck --retry {r['id']} --reset-revisions")
        print(f"    reject  : python -m livingbook.cli reject {r['id']} --reason '...'")
    print()
    return 0


# ── argument parsing ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livingbook",
        description="Autonomous research-to-publication pipeline for a living NLP textbook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--version", action="version", version=f"livingbook {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check environment, contracts and credentials")

    idx = sub.add_parser("index", help="build the book knowledge base")
    idx.add_argument("--force", action="store_true", help="re-summarise everything")
    idx.add_argument("--structure-only", action="store_true")
    idx.add_argument("--concurrency", type=int, default=8)
    idx.add_argument("--limit", type=int, default=None)

    sub.add_parser("discover", help="run one research discovery pass")
    sub.add_parser("synthesize", help="cluster undigested research")

    cyc = sub.add_parser("cycle", help="one full research-to-publication cycle")
    cyc.add_argument("--no-discover", action="store_true")
    cyc.add_argument("--no-drive", action="store_true")

    sub.add_parser("tick", help="advance every active pipeline one step")

    dmn = sub.add_parser("daemon", help="run continuously on the configured schedule")
    dmn.add_argument("--reset-schedule", action="store_true",
                     help="clear stale locks and make every job due now")

    sub.add_parser("status", help="show what the system is doing")

    apr = sub.add_parser("approve", help="approve a pipeline awaiting human review")
    apr.add_argument("pipeline_id")
    apr.add_argument("--by", default="human")
    apr.add_argument("--run", action="store_true", help="drive it to completion after approving")

    rej = sub.add_parser("reject", help="reject a pipeline awaiting human review")
    rej.add_argument("pipeline_id")
    rej.add_argument("--reason", default="")

    qa = sub.add_parser("qa", help="run full-book QA")
    qa.add_argument("--no-global", action="store_true")
    qa.add_argument("--no-build", action="store_true")
    qa.add_argument("--links", action="store_true", help="also check external URLs")

    llm = sub.add_parser("llm", help="LLM provider operations")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    llm_sub.add_parser("status")
    llm_sub.add_parser("probe", help="re-measure which models actually respond")
    llm_sub.add_parser("revive", help="re-enable disabled keys")
    llm_sub.add_parser("models", help="list models the pool can reach")
    keys = llm_sub.add_parser("keys", help="per-key usage")
    keys.add_argument("--limit", type=int, default=30)

    kb = sub.add_parser("kb", help="query the book knowledge base")
    kb_sub = kb.add_subparsers(dest="kb_command", required=True)
    out = kb_sub.add_parser("outline")
    out.add_argument("--depth", default="section",
                     choices=["part", "chapter", "section", "subsection"])
    out.add_argument("--no-summaries", action="store_true")
    srch = kb_sub.add_parser("search")
    srch.add_argument("query")
    srch.add_argument("--limit", type=int, default=8)
    imp = kb_sub.add_parser("impact", help="which sections new research would affect")
    imp.add_argument("concepts", nargs="+")
    imp.add_argument("--limit", type=int, default=10)
    cpt = kb_sub.add_parser("concept")
    cpt.add_argument("name")
    cit = kb_sub.add_parser("citation")
    cit.add_argument("key")
    kb_sub.add_parser("stats")
    dup = kb_sub.add_parser("duplication")
    dup.add_argument("--min-nodes", type=int, default=4)

    git = sub.add_parser("git", help="repository operations")
    git_sub = git.add_subparsers(dest="git_command", required=True)
    git_sub.add_parser("init", help="initialise and attach the Living Book remote")
    git_sub.add_parser("status")
    glog = git_sub.add_parser("log")
    glog.add_argument("--limit", type=int, default=20)

    tr = sub.add_parser("trace", help="trace a run, a research item or a pipeline")
    tr.add_argument("--run-id")
    tr.add_argument("--research-id")
    tr.add_argument("--pipeline-id")
    tr.add_argument("--limit", type=int, default=200)

    vis = sub.add_parser("visual", help="resolve outstanding figure placeholders")
    vis.add_argument("--key", help="resolve one specific \\bookimage key")
    vis.add_argument("--limit", type=int, default=3)

    em = sub.add_parser("email", help="email operations")
    em_sub = em.add_subparsers(dest="email_command", required=True)
    em_sub.add_parser("status")
    em_sub.add_parser("test", help="send a test message to the configured recipients")

    st = sub.add_parser("stuck", help="pipelines needing an operator; retry or inspect")
    st.add_argument("--retry", metavar="PIPELINE_ID", help="put this pipeline back on the path")
    st.add_argument("--from-state", help="state to resume from (default: where it was)")
    st.add_argument("--reset-revisions", action="store_true",
                    help="clear the revision count, for when the system was at fault")
    st.add_argument("--note", default="", help="why it is being retried")
    st.add_argument("--run", action="store_true", help="drive it after retrying")
    st.add_argument("--resume-all", action="store_true", help="resume every FAILED pipeline")

    mcpp = sub.add_parser("mcp", help="Model Context Protocol server and client")
    mcp_sub = mcpp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("serve", help="run the MCP server on stdio (clients spawn this)")
    mcp_sub.add_parser("tools", help="list what the server exposes")
    mcp_sub.add_parser("external", help="connect configured external MCP servers")

    return p


COMMANDS = {
    "doctor": cmd_doctor, "index": cmd_index, "discover": cmd_discover,
    "synthesize": cmd_synthesize, "cycle": cmd_cycle, "tick": cmd_tick,
    "daemon": cmd_daemon, "status": cmd_status, "approve": cmd_approve,
    "reject": cmd_reject, "qa": cmd_qa, "llm": cmd_llm, "kb": cmd_kb,
    "git": cmd_git, "trace": cmd_trace, "visual": cmd_visual, "email": cmd_email,
    "mcp": cmd_mcp, "stuck": cmd_stuck,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
