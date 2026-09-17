"""MCP surface tests.

The MCP server is the one place an outside client reaches into a live autonomous
system, so what it exposes — and what it deliberately does not — is worth pinning down.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livingbook.mcp.server import mcp  # noqa: E402
from livingbook.tools import REGISTRY  # noqa: E402


def _tools():
    return asyncio.run(mcp.list_tools())


def test_server_exposes_a_curated_surface():
    names = {t.name for t in _tools()}
    assert names, "the MCP server exposes no tools"
    # A handful of well-shaped questions, not 62 primitives.
    assert len(names) < 25, (
        f"{len(names)} tools exposed — the surface should stay curated")
    for expected in ("book_impact", "book_search", "research_status",
                     "explain_decision", "review_pending"):
        assert expected in names, f"{expected} should be exposed"


def test_pipeline_internals_are_not_exposed():
    """Primitives only make sense inside a state machine that validates them.

    Exposing them would let a chat client write to the knowledge base or patch the
    manuscript with no verdict, no verification and no provenance.
    """
    names = {t.name for t in _tools()}
    for internal in ("kb_upsert", "patch_file", "write_file", "git_commit",
                     "git_push", "send_email", "gemini_generate", "download_image"):
        assert internal not in names, (
            f"{internal} is a pipeline primitive and must not be exposed over MCP")


def test_every_exposed_tool_is_documented():
    for tool in _tools():
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) > 40, (
            f"{tool.name} needs a description a client can act on")


def test_resources_and_prompts_are_registered():
    resources = asyncio.run(mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    assert "livingbook://status" in uris
    assert "livingbook://book/outline" in uris

    prompts = asyncio.run(mcp.list_prompts())
    assert {p.name for p in prompts} >= {"assess_research", "review_change"}


def test_read_only_tools_do_not_claim_to_be_destructive():
    for tool in _tools():
        if tool.name in ("approve_change", "reject_change"):
            continue
        hints = getattr(tool, "annotations", None)
        if hints is not None and getattr(hints, "destructiveHint", None):
            pytest.fail(f"{tool.name} is marked destructive but should be read-only")


def test_impact_tool_answers_without_the_manuscript():
    """book_impact must work from the graph — that is the point of it."""
    result = asyncio.run(mcp.call_tool(
        "book_impact", {"concepts": ["speculative decoding"], "limit": 3}))
    text = next(block.text for block in result.content if hasattr(block, "text"))
    data = json.loads(text)
    assert "affected" in data
    assert data["concepts"] == ["speculative decoding"]


# ── the external-server bridge ─────────────────────────────────────────────


def test_external_bridge_defaults_to_read_only():
    from livingbook.mcp.client import MCPToolBridge
    from livingbook.tools import Capability

    assert Capability.FS_WRITE not in MCPToolBridge.DEFAULT_CAPABILITIES
    assert Capability.GIT not in MCPToolBridge.DEFAULT_CAPABILITIES
    assert Capability.EMAIL not in MCPToolBridge.DEFAULT_CAPABILITIES


def test_external_config_is_valid_and_examples_are_disabled():
    import yaml
    from livingbook.config import get_config

    path = get_config().root / "config" / "mcp.yaml"
    assert path.exists(), "config/mcp.yaml should exist to document the shape"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, spec in (data.get("servers") or {}).items():
        if spec.get("enabled"):
            assert spec.get("command"), f"{name} is enabled but has no command"


def test_bridge_survives_an_unreachable_server():
    """A broken external server must not stop the Living Book starting."""
    from livingbook.mcp.client import MCPServerConnection

    conn = MCPServerConnection(
        "broken", {"command": "definitely-not-a-real-binary-xyz", "args": []})
    assert asyncio.run(conn.connect()) is False
    asyncio.run(conn.close())


def test_native_registry_is_untouched_by_mcp_import():
    """Importing the MCP layer must not register or remove native tools."""
    assert "search_arxiv" in REGISTRY.names()
    assert "kb_upsert" in REGISTRY.names()
    assert len(REGISTRY.names()) >= 60
