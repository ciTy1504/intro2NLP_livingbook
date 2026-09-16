"""Tool layer.

Importing this package registers every tool. ``AgentContext`` is the only sanctioned
way to invoke one — it enforces the per-agent allowlist, the capability grants and the
write-path scoping declared in ``config/agents.yaml``.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for registration side effects)
    book,
    code_hosting,
    community,
    email_tools,
    filesystem,
    git_tools,
    kb,
    llm_tools,
    scholarly,
    visual,
    web,
)
from .registry import (
    REGISTRY,
    AgentContext,
    Capability,
    PermissionDenied,
    ToolError,
    ToolRegistry,
    ToolUnavailable,
    system_context,
    tool,
)

__all__ = [
    "REGISTRY", "AgentContext", "Capability", "PermissionDenied",
    "ToolError", "ToolRegistry", "ToolUnavailable", "system_context", "tool",
]


def tool_names() -> list[str]:
    return REGISTRY.names()


def describe_tools() -> list[dict]:
    return REGISTRY.describe()
