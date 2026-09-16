"""Tool registry, capabilities and permission enforcement.

A tool is a primitive: it does one thing, does no reasoning, and returns plain data.
Every tool declares the capabilities it needs, and every call goes through
``AgentContext.call()``, which:

  1. checks the tool is in the calling agent's allowlist,
  2. checks the tool's capabilities are a subset of the agent's grants,
  3. checks any path constraint the agent carries,
  4. times the call and emits a structured event.

Capability checks happen here rather than inside each tool so that a new tool cannot
accidentally ship without enforcement, and so the permission matrix in
``config/agents.yaml`` is the single source of truth.
"""

from __future__ import annotations

import fnmatch
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from ..config import get_config
from ..obs import get_logger


class Capability(str, Enum):
    SEARCH = "SEARCH"        # query an external index (may be rate limited / metered)
    FETCH = "FETCH"          # retrieve a specific external resource by URL/id
    SCHOLARLY = "SCHOLARLY"  # bibliographic databases and DOI resolution
    FS_READ = "FS_READ"
    FS_WRITE = "FS_WRITE"
    KB_READ = "KB_READ"
    KB_WRITE = "KB_WRITE"
    LLM = "LLM"
    BUILD = "BUILD"          # run the LaTeX build / test suite
    GIT = "GIT"
    EMAIL = "EMAIL"


class PermissionDenied(RuntimeError):
    """An agent attempted a tool it is not granted."""


class ToolError(RuntimeError):
    """A tool failed in a way the caller may be able to handle."""


class ToolUnavailable(ToolError):
    """An external source is down or not configured — callers should degrade."""


@dataclass(slots=True)
class ToolSpec:
    name: str
    fn: Callable[..., Awaitable[Any]]
    capabilities: frozenset[Capability]
    description: str
    destructive: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(
                f"unknown tool {name!r}; registered: {sorted(self._tools)[:12]}…"
            )
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "capabilities": sorted(c.value for c in s.capabilities),
                "description": s.description,
            }
            for s in sorted(self._tools.values(), key=lambda s: s.name)
        ]

    def by_capability(self, cap: Capability) -> list[str]:
        return sorted(n for n, s in self._tools.items() if cap in s.capabilities)


REGISTRY = ToolRegistry()


def tool(
    name: str,
    capabilities: Iterable[Capability | str],
    *,
    description: str = "",
    destructive: bool = False,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Register an async function as a tool."""

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"tool {name!r} must be async")
        caps = frozenset(
            c if isinstance(c, Capability) else Capability(str(c)) for c in capabilities
        )
        REGISTRY.register(ToolSpec(
            name=name, fn=fn, capabilities=caps,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            destructive=destructive,
        ))
        fn._tool_name = name  # type: ignore[attr-defined]
        return fn

    return decorator


@dataclass
class AgentContext:
    """The handle an agent uses to reach everything outside itself.

    Carries identity (for logging), grants (for enforcement) and constraints (for
    write scoping). An agent never imports a tool module directly.
    """

    agent: str
    capabilities: frozenset[Capability]
    allowed_tools: frozenset[str]
    constraints: dict[str, Any] = field(default_factory=dict)
    pipeline_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_agent(cls, agent_name: str, *, pipeline_id: str | None = None) -> "AgentContext":
        spec = get_config().agent_spec(agent_name)
        caps = frozenset(Capability(c) for c in spec.get("capabilities", []))
        tools = frozenset(spec.get("tools", []))
        return cls(
            agent=agent_name,
            capabilities=caps,
            allowed_tools=tools,
            constraints=dict(spec.get("constraints", {}) or {}),
            pipeline_id=pipeline_id,
            extra={"model_role": spec.get("model_role", "balanced"),
                   "failure_policy": spec.get("failure_policy", "retry"),
                   "evidence_kinds": spec.get("evidence_kinds", [])},
        )

    # -- enforcement -------------------------------------------------------
    def check(self, tool_name: str) -> ToolSpec:
        spec = REGISTRY.get(tool_name)
        if tool_name not in self.allowed_tools:
            raise PermissionDenied(
                f"agent {self.agent!r} is not allowed tool {tool_name!r} "
                f"(allowlist in config/agents.yaml)"
            )
        missing = spec.capabilities - self.capabilities
        if missing:
            raise PermissionDenied(
                f"agent {self.agent!r} lacks capability "
                f"{sorted(c.value for c in missing)} required by {tool_name!r}"
            )
        return spec

    def check_write_path(self, path: str | Path) -> Path:
        """Enforce the agent's write scope.

        Write scoping is what stops, say, the image search agent from writing into
        manuscript/ — the permission matrix is only meaningful if paths are checked
        too, since FS_WRITE alone would otherwise mean "write anywhere".
        """
        # An agent without FS_WRITE must not pass a write check at all. Without this,
        # a read-only agent that reached this helper directly (rather than through a
        # tool call) would be scoped only by `write_paths` — and an agent with no
        # `write_paths` declared has no scope, so it would be allowed anywhere.
        if Capability.FS_WRITE not in self.capabilities:
            raise PermissionDenied(
                f"agent {self.agent!r} has no FS_WRITE capability and may not write "
                f"{path}"
            )

        root = get_config().root
        target = Path(path)
        resolved = (root / target).resolve() if not target.is_absolute() else target.resolve()

        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            raise PermissionDenied(
                f"agent {self.agent!r} may not write outside the repository: {resolved}"
            )
        rel_posix = rel.as_posix()

        forbidden = self.constraints.get("forbidden_paths") or []
        for pattern in forbidden:
            if fnmatch.fnmatch(rel_posix, pattern):
                raise PermissionDenied(
                    f"agent {self.agent!r} is forbidden from writing {rel_posix} "
                    f"(matches {pattern!r})"
                )

        allowed = self.constraints.get("write_paths")
        if allowed:
            if not any(fnmatch.fnmatch(rel_posix, pattern) for pattern in allowed):
                raise PermissionDenied(
                    f"agent {self.agent!r} may only write {allowed}; refused {rel_posix}"
                )
        return resolved

    # -- invocation --------------------------------------------------------
    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        spec = self.check(tool_name)
        log = get_logger()
        start = time.monotonic()
        try:
            if _wants_context(spec.fn):
                result = await spec.fn(self, **kwargs)
            else:
                result = await spec.fn(**kwargs)
        except PermissionDenied:
            raise
        except ToolUnavailable as exc:
            log.warn(f"{tool_name} unavailable: {exc}", tool=tool_name, status="unavailable",
                     duration_ms=int((time.monotonic() - start) * 1000))
            raise
        except Exception as exc:
            log.error(f"{tool_name} failed: {type(exc).__name__}: {exc}",
                      tool=tool_name, status="error",
                      duration_ms=int((time.monotonic() - start) * 1000))
            raise
        log.debug(f"{tool_name} ok", tool=tool_name, status="ok",
                  duration_ms=int((time.monotonic() - start) * 1000))
        return result

    async def try_call(self, tool_name: str, *, default: Any = None, **kwargs: Any) -> Any:
        """Call a tool, degrading to ``default`` if the source is unavailable.

        Used by research agents so one dead source never fails a discovery cycle.
        """
        try:
            return await self.call(tool_name, **kwargs)
        except (ToolUnavailable, ToolError) as exc:
            get_logger().warn(f"{tool_name} degraded: {exc}", tool=tool_name, status="degraded")
            return default
        except PermissionDenied:
            raise
        except Exception as exc:
            get_logger().warn(f"{tool_name} degraded: {type(exc).__name__}: {exc}",
                              tool=tool_name, status="degraded")
            return default

    @property
    def model_role(self) -> str:
        return str(self.extra.get("model_role", "balanced"))


def _wants_context(fn: Callable[..., Any]) -> bool:
    params = list(inspect.signature(fn).parameters)
    return bool(params) and params[0] in ("ctx", "context")


def system_context(agent: str = "system") -> AgentContext:
    """An unrestricted context for setup scripts and the CLI, not for agents."""
    return AgentContext(
        agent=agent,
        capabilities=frozenset(Capability),
        allowed_tools=frozenset(REGISTRY.names()),
    )
