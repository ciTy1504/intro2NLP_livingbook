"""Agent base class.

An agent is a declared role, not a prompt. Everything that defines it — objective,
capabilities, tool allowlist, constraints, model tier, failure policy — comes from
``config/agents.yaml``, so the permission matrix in the architecture is the single
source of truth rather than a description of what the code happens to do.

The failure policy is applied here, uniformly:

    retry               transient failures, bounded by max_retries
    degrade             return a partial/empty result; the caller continues
    fail_pipeline       raise; the pipeline moves to FAILED
    escalate_to_human   raise a distinguishable error; the pipeline parks in NEEDS_HUMAN
"""

from __future__ import annotations

import abc
import asyncio
import random
from typing import Any, Generic, TypeVar

from ..config import get_config
from ..obs import agent_scope, get_logger
from ..skills import Skill, get_skill
from ..state.artifacts import ArtifactStore, get_artifacts
from ..state.store import Store, get_store
from ..tools import AgentContext, PermissionDenied, ToolError, ToolUnavailable

TOut = TypeVar("TOut")


class AgentFailure(RuntimeError):
    """The agent could not complete its objective."""


class EscalateToHuman(AgentFailure):
    """The agent stopped and needs a person. The pipeline parks rather than fails."""


class AgentDegraded(RuntimeError):
    """A degraded agent produced nothing usable, but the cycle may continue."""


class BaseAgent(abc.ABC, Generic[TOut]):
    #: Must match a key under `agents:` in config/agents.yaml.
    name: str = "agent"

    def __init__(self, *, pipeline_id: str | None = None) -> None:
        self.cfg = get_config()
        self.spec = self.cfg.agent_spec(self.name)
        self.ctx = AgentContext.for_agent(self.name, pipeline_id=pipeline_id)
        self.log = get_logger()
        self.store: Store = get_store()
        self.artifacts: ArtifactStore = get_artifacts()
        self.pipeline_id = pipeline_id
        self._skills: dict[str, Skill] = {}

    # -- declared contract -------------------------------------------------
    @property
    def objective(self) -> str:
        return str(self.spec.get("objective", "")).strip()

    @property
    def failure_policy(self) -> str:
        return str(self.spec.get("failure_policy", "retry"))

    @property
    def max_retries(self) -> int:
        return int(self.spec.get("max_retries", 3))

    @property
    def model_role(self) -> str:
        return str(self.spec.get("model_role", "balanced"))

    @property
    def allowed_skills(self) -> list[str]:
        return list(self.spec.get("skills", []) or [])

    def skill(self, name: str) -> Skill:
        """Resolve a skill, refusing any the agent's contract does not grant."""
        if name not in self.allowed_skills:
            raise PermissionDenied(
                f"agent {self.name!r} is not granted skill {name!r} "
                f"(its skills are {self.allowed_skills})"
            )
        if name not in self._skills:
            self._skills[name] = get_skill(name)
        return self._skills[name]

    # -- execution ---------------------------------------------------------
    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> TOut:
        """The agent's actual work. Subclasses implement this, callers use run()."""

    async def run(self, **kwargs: Any) -> TOut:
        """Execute under this agent's attribution and failure policy."""
        policy = self.failure_policy
        attempts = self.max_retries if policy == "retry" else 1
        last: Exception | None = None

        with agent_scope(self.name):
            for attempt in range(1, attempts + 1):
                try:
                    return await self.execute(**kwargs)
                except PermissionDenied:
                    # A permission error is a configuration bug. Retrying it just
                    # delays the fix and hides the cause.
                    raise
                except EscalateToHuman:
                    raise
                except (ToolUnavailable, ToolError, AgentDegraded) as exc:
                    last = exc
                    if policy == "degrade":
                        self.log.warn(
                            f"{self.name} degraded: {type(exc).__name__}: {exc}",
                            status="degraded")
                        return self.degraded_result()
                    if attempt < attempts:
                        delay = min(2.0 * attempt, 12.0) * (0.5 + random.random())
                        self.log.warn(
                            f"{self.name} attempt {attempt}/{attempts} failed "
                            f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                except Exception as exc:
                    last = exc
                    if attempt < attempts:
                        delay = min(2.0 * attempt, 12.0) * (0.5 + random.random())
                        self.log.warn(
                            f"{self.name} attempt {attempt}/{attempts} failed "
                            f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)

        message = f"{self.name} failed after {attempts} attempt(s): {last}"
        if policy == "degrade":
            self.log.warn(message, status="degraded")
            return self.degraded_result()
        if policy == "escalate_to_human":
            raise EscalateToHuman(message) from last
        raise AgentFailure(message) from last

    def degraded_result(self) -> TOut:
        """What a degraded agent returns. Override where an empty list is wrong."""
        return []  # type: ignore[return-value]

    # -- artifacts ---------------------------------------------------------
    def save(self, kind: str, payload: Any, *, parent: str | None = None,
             meta: dict[str, Any] | None = None) -> str:
        return self.artifacts.write(
            kind, payload, pipeline_id=self.pipeline_id, parent=parent, meta=meta)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def validate_agent_registry(agent_classes: dict[str, type[BaseAgent]]) -> list[str]:
    """Check every agent's declared skills and tools actually exist.

    Run by the CLI's `doctor` command, because a typo in agents.yaml would otherwise
    only surface hours into an unattended run.
    """
    from ..skills import SKILLS
    from ..tools import REGISTRY, Capability

    cfg = get_config()
    problems: list[str] = []

    for name in cfg.all_agent_names():
        spec = cfg.agent_spec(name)
        caps = set()
        for c in spec.get("capabilities", []) or []:
            try:
                caps.add(Capability(c))
            except ValueError:
                problems.append(f"{name}: unknown capability {c!r}")

        for skill_name in spec.get("skills", []) or []:
            if skill_name not in SKILLS:
                problems.append(f"{name}: unknown skill {skill_name!r}")

        for tool_name in spec.get("tools", []) or []:
            try:
                tool_spec = REGISTRY.get(tool_name)
            except KeyError:
                problems.append(f"{name}: unknown tool {tool_name!r}")
                continue
            missing = tool_spec.capabilities - caps
            if missing:
                problems.append(
                    f"{name}: tool {tool_name!r} requires "
                    f"{sorted(c.value for c in missing)} which the agent is not granted")

        if name in agent_classes:
            declared = set(spec.get("skills", []) or [])
            for skill_name in getattr(agent_classes[name], "uses_skills", ()):
                if skill_name not in declared:
                    problems.append(
                        f"{name}: implementation uses skill {skill_name!r} "
                        f"which its contract does not grant")

    for name in agent_classes:
        if name not in cfg.all_agent_names():
            problems.append(f"{name}: implemented but absent from config/agents.yaml")

    return problems
