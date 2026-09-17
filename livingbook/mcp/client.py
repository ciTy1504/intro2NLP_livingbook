"""MCP client — mount external MCP servers as Living Book tools.

This makes the tool layer extensible without code changes: a server listed in
``config/mcp.yaml`` has its tools discovered at startup and registered alongside the
native ones, subject to the same capability tags and per-agent allowlists.

An honest caveat about when to use it. The system already has 62 native adapters tuned
per source — the arXiv adapter knows that an author's speedup claim is an
``author_claim`` and not a result, the GitHub adapter knows a star count is adoption
evidence rather than scientific evidence. A generic MCP tool returns text with none of
that structure, so routing arXiv through an MCP server would be a downgrade. Where this
earns its place is reaching things the system has no adapter for and never will: a
private paper database, a team wiki, an internal benchmark service.

Capabilities are assigned per server in config, not inferred, because a tool's name
tells you nothing about whether it writes. An unclassified server gets the read-only
default.
"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import AsyncExitStack
from typing import Any

from ..config import get_config
from ..obs import get_logger
from ..tools.registry import REGISTRY, Capability, ToolSpec, ToolUnavailable


class MCPServerConnection:
    """One external MCP server, connected over stdio and kept alive."""

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        self.name = name
        self.spec = spec
        self.log = get_logger()
        self.session: Any = None
        self.tools: list[Any] = []
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> bool:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = self.spec.get("command")
        if not command:
            self.log.warn(f"mcp/{self.name}: no command configured")
            return False

        resolved = shutil.which(command) or command
        params = StdioServerParameters(
            command=resolved,
            args=list(self.spec.get("args", []) or []),
            env=dict(self.spec.get("env", {}) or {}) or None,
        )

        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self.session = await self._stack.enter_async_context(
                ClientSession(read, write))
            await self.session.initialize()
            listed = await self.session.list_tools()
            self.tools = list(listed.tools)
        except Exception as exc:
            # A broken external server must not stop the Living Book starting.
            self.log.warn(
                f"mcp/{self.name}: could not connect ({type(exc).__name__}: {exc})")
            await self.close()
            return False

        self.log.info(
            f"mcp/{self.name}: connected, {len(self.tools)} tool(s): "
            + ", ".join(t.name for t in self.tools[:8]))
        return True

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise ToolUnavailable(f"mcp/{self.name} is not connected")
        result = await self.session.call_tool(tool_name, arguments)

        if getattr(result, "isError", False):
            raise ToolUnavailable(f"mcp/{self.name}:{tool_name} returned an error")

        # Prefer structured content; fall back to concatenated text blocks.
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
        self.session = None


class MCPToolBridge:
    """Discovers external MCP servers and registers their tools natively."""

    #: Read-only by default. A server that writes must say so in config, because
    #: nothing in a tool's name reliably indicates whether it has side effects.
    DEFAULT_CAPABILITIES = (Capability.FETCH,)

    def __init__(self) -> None:
        self.cfg = get_config()
        self.log = get_logger()
        self.connections: dict[str, MCPServerConnection] = {}
        self.registered: list[str] = []

    def _servers(self) -> dict[str, dict[str, Any]]:
        path = self.cfg.root / "config" / "mcp.yaml"
        if not path.exists():
            return {}
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            name: spec for name, spec in (data.get("servers") or {}).items()
            if spec and spec.get("enabled", True)
        }

    async def connect_all(self) -> dict[str, int]:
        """Connect every configured server and register its tools. Never raises."""
        servers = self._servers()
        if not servers:
            self.log.debug("mcp: no external servers configured")
            return {}

        summary: dict[str, int] = {}
        for name, spec in servers.items():
            conn = MCPServerConnection(name, spec)
            if not await conn.connect():
                summary[name] = 0
                continue
            self.connections[name] = conn
            summary[name] = self._register(conn, spec)
        return summary

    def _register(self, conn: MCPServerConnection, spec: dict[str, Any]) -> int:
        caps = frozenset(
            Capability(c) for c in (spec.get("capabilities") or [])
        ) or frozenset(self.DEFAULT_CAPABILITIES)
        prefix = spec.get("prefix", f"mcp_{conn.name}")
        allow = set(spec.get("only_tools") or [])

        count = 0
        for tool in conn.tools:
            if allow and tool.name not in allow:
                continue
            registered_name = f"{prefix}_{tool.name}"
            if registered_name in REGISTRY.names():
                continue

            async def _invoke(_conn=conn, _tool=tool.name, **kwargs: Any) -> Any:
                return await _conn.call(_tool, kwargs)

            REGISTRY.register(ToolSpec(
                name=registered_name,
                fn=_invoke,
                capabilities=caps,
                description=(tool.description or f"{conn.name}: {tool.name}")[:300],
            ))
            self.registered.append(registered_name)
            count += 1
        return count

    async def close_all(self) -> None:
        for conn in self.connections.values():
            await conn.close()
        self.connections.clear()

    def status(self) -> dict[str, Any]:
        return {
            "servers": {
                name: {
                    "connected": conn.session is not None,
                    "tools": [t.name for t in conn.tools],
                }
                for name, conn in self.connections.items()
            },
            "registered_tools": self.registered,
        }


_bridge: MCPToolBridge | None = None


async def mount_external_servers() -> dict[str, int]:
    """Connect configured MCP servers. Called once at orchestrator startup."""
    global _bridge
    if _bridge is None:
        _bridge = MCPToolBridge()
        return await _bridge.connect_all()
    return {}


def get_bridge() -> MCPToolBridge | None:
    return _bridge


async def probe_servers() -> dict[str, Any]:
    """Connect, list what each server offers, disconnect. For the CLI."""
    bridge = MCPToolBridge()
    summary = await bridge.connect_all()
    status = bridge.status()
    await bridge.close_all()
    return {"tools_registered": summary, **status}
