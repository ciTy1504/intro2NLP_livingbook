"""Skill base class.

A skill is a reusable workflow: prompt templates, tool orchestration and output
parsing. Skills are stateless and receive an ``AgentContext``, so the same skill run by
two different agents inherits each agent's permissions — the `book_retrieval` skill
used by the Writer can read the manuscript, and used by the Citation Finder it cannot,
without the skill containing any knowledge of who is calling it.
"""

from __future__ import annotations

import abc
from typing import Any

from ..obs import get_logger, skill_scope
from ..tools import AgentContext


class Skill(abc.ABC):
    name: str = "skill"
    #: Tools this skill may use. Checked against the calling agent's grants so a
    #: misconfiguration surfaces as a clear message rather than a mid-run denial.
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.log = get_logger()

    async def __call__(self, ctx: AgentContext, **kwargs: Any) -> Any:
        self.check_permissions(ctx)
        with skill_scope(self.name):
            return await self.run(ctx, **kwargs)

    @abc.abstractmethod
    async def run(self, ctx: AgentContext, **kwargs: Any) -> Any:
        ...

    def check_permissions(self, ctx: AgentContext) -> None:
        missing = [t for t in self.required_tools if t not in ctx.allowed_tools]
        if missing:
            raise PermissionError(
                f"agent {ctx.agent!r} cannot run skill {self.name!r}: "
                f"missing required tools {missing}. Add them in config/agents.yaml."
            )

    def available(self, ctx: AgentContext, tool_name: str) -> bool:
        return tool_name in ctx.allowed_tools


def json_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build a Gemini responseSchema object."""
    return {"type": "OBJECT", "properties": properties, "required": required}


def array_of(item: dict[str, Any], description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "ARRAY", "items": item}
    if description:
        out["description"] = description
    return out


def string(description: str = "", enum: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "STRING"}
    if description:
        out["description"] = description
    if enum:
        out["enum"] = enum
    return out


def boolean(description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "BOOLEAN"}
    if description:
        out["description"] = description
    return out


def number(description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "NUMBER"}
    if description:
        out["description"] = description
    return out


def integer(description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "INTEGER"}
    if description:
        out["description"] = description
    return out


def obj(properties: dict[str, Any], required: list[str] | None = None,
        description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "OBJECT", "properties": properties}
    if required:
        out["required"] = required
    if description:
        out["description"] = description
    return out
