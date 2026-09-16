"""Permission model tests.

The permission matrix is only meaningful if it is enforced, so these assert the
specific denials the architecture depends on — particularly that the Research
Synthesizer cannot reach the network and that only the Asset Manager can write into
the manuscript's image directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.agents import AGENTS, validate_all  # noqa: E402
from livingbook.config import get_config  # noqa: E402
from livingbook.skills import SKILLS  # noqa: E402
from livingbook.tools import (  # noqa: E402
    REGISTRY,
    AgentContext,
    Capability,
    PermissionDenied,
)


def test_all_agent_contracts_are_valid():
    problems = validate_all()
    assert not problems, "agent contract problems:\n  " + "\n  ".join(problems)


def test_every_configured_agent_is_implemented():
    configured = set(get_config().all_agent_names()) - {"orchestrator"}
    assert configured <= set(AGENTS), (
        f"configured but not implemented: {sorted(configured - set(AGENTS))}")


def test_every_agent_tool_exists_and_is_granted():
    cfg = get_config()
    for name in cfg.all_agent_names():
        spec = cfg.agent_spec(name)
        caps = {Capability(c) for c in spec.get("capabilities", [])}
        for tool_name in spec.get("tools", []):
            tool = REGISTRY.get(tool_name)
            assert tool.capabilities <= caps, (
                f"{name} lists {tool_name} but lacks "
                f"{sorted(c.value for c in tool.capabilities - caps)}")


def test_every_agent_skill_exists():
    cfg = get_config()
    for name in cfg.all_agent_names():
        for skill in cfg.agent_spec(name).get("skills", []) or []:
            assert skill in SKILLS, f"{name} declares unknown skill {skill!r}"


def test_synthesizer_cannot_reach_the_network():
    """The synthesizer reasons only over collected evidence — that is the whole point."""
    ctx = AgentContext.for_agent("research_synthesizer")
    assert Capability.SEARCH not in ctx.capabilities
    assert Capability.FETCH not in ctx.capabilities
    for tool_name in ("search_arxiv", "web_search", "fetch_url", "search_github"):
        with pytest.raises(PermissionDenied):
            ctx.check(tool_name)


def test_orchestrator_has_no_llm_access():
    """It schedules work; it does not form opinions about research."""
    ctx = AgentContext.for_agent("orchestrator")
    assert Capability.LLM not in ctx.capabilities


def test_writer_write_scope():
    ctx = AgentContext.for_agent("writer_agent")
    ctx.check_write_path("manuscript/part1/chapters_part1/chap1_1.tex")
    for forbidden in ("manuscript/references.bib", "manuscript/main.tex",
                      "manuscript/style.tex", "config/config.yaml",
                      "secrets/.env", "../outside.txt"):
        with pytest.raises(PermissionDenied):
            ctx.check_write_path(forbidden)


def test_only_asset_manager_writes_manuscript_images():
    """A single choke point is what makes the licence gate enforceable."""
    AgentContext.for_agent("asset_manager").check_write_path("manuscript/images/x.png")
    for agent in ("image_search_agent", "diagram_generator_agent", "writer_agent"):
        with pytest.raises(PermissionDenied):
            AgentContext.for_agent(agent).check_write_path("manuscript/images/x.png")


def test_only_bibtex_validator_writes_the_bibliography():
    AgentContext.for_agent("bibtex_validator").check_write_path("manuscript/references.bib")
    for agent in ("writer_agent", "citation_finder", "asset_manager"):
        ctx = AgentContext.for_agent(agent)
        with pytest.raises(PermissionDenied):
            ctx.check_write_path("manuscript/references.bib")


def test_image_agents_confined_to_their_directories():
    AgentContext.for_agent("image_search_agent").check_write_path(
        "figures/candidates/a.png")
    AgentContext.for_agent("diagram_generator_agent").check_write_path(
        "figures/generated/a.png")
    with pytest.raises(PermissionDenied):
        AgentContext.for_agent("image_search_agent").check_write_path(
            "figures/generated/a.png")


def test_only_delivery_agents_hold_git_and_email():
    for name in get_config().all_agent_names():
        ctx = AgentContext.for_agent(name)
        if Capability.GIT in ctx.capabilities:
            assert name == "git_agent", f"{name} should not have GIT"
        if Capability.EMAIL in ctx.capabilities:
            assert name == "email_agent", f"{name} should not have EMAIL"


def test_research_agents_cannot_write_files():
    for name in ("arxiv_agent", "scholarly_agent", "conference_agent",
                 "community_agent", "benchmark_agent", "research_blog_agent"):
        ctx = AgentContext.for_agent(name)
        assert Capability.FS_WRITE not in ctx.capabilities, f"{name} must not write files"


def test_verifiers_cannot_write_the_manuscript():
    for name in ("technical_verifier", "editorial_verifier", "book_qa_agent",
                 "cross_chapter_consistency_agent", "citation_auditor"):
        ctx = AgentContext.for_agent(name)
        assert Capability.FS_WRITE not in ctx.capabilities, (
            f"{name} verifies; it must not edit")
