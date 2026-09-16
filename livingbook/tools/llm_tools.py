"""LLM tools.

Thin wrappers over the provider layer so that LLM use is subject to the same
permission model and the same structured logging as every other tool. Agents never
import the provider directly — that is what keeps "only the provider layer touches
keys" true rather than aspirational.

The ``role`` defaults to the calling agent's configured ``model_role``, so an agent's
model tier is a property of its contract in ``config/agents.yaml`` rather than
something each call site decides.
"""

from __future__ import annotations

from typing import Any

from ..llm import GenerationOptions, get_provider
from .registry import AgentContext, Capability, ToolError, tool


@tool("gemini_generate", [Capability.LLM],
      description="Generate free text with the LLM provider.")
async def gemini_generate(
    ctx: AgentContext,
    prompt: str,
    *,
    role: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    cache: bool = True,
) -> dict[str, Any]:
    provider = get_provider()
    resp = await provider.generate(
        prompt,
        role=role or ctx.model_role,
        options=GenerationOptions(
            temperature=temperature, max_output_tokens=max_output_tokens,
            system_instruction=system, cache=cache,
        ),
    )
    return {
        "text": resp.text, "model": resp.model, "cached": resp.cached,
        "tokens": resp.usage.total_tokens, "finish_reason": resp.finish_reason,
    }


@tool("gemini_structured_output", [Capability.LLM],
      description="Generate JSON conforming to a schema. The default for agent output.")
async def gemini_structured_output(
    ctx: AgentContext,
    prompt: str,
    schema: dict[str, Any],
    *,
    role: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    cache: bool = True,
) -> dict[str, Any]:
    if not isinstance(schema, dict) or "type" not in schema:
        raise ToolError("schema must be a dict with a 'type' key")
    provider = get_provider()
    resp = await provider.generate_structured(
        prompt, schema,
        role=role or ctx.model_role,
        options=GenerationOptions(
            temperature=temperature, max_output_tokens=max_output_tokens,
            system_instruction=system, cache=cache,
        ),
    )
    return {
        "data": resp.data, "model": resp.model, "cached": resp.cached,
        "tokens": resp.usage.total_tokens, "repaired": resp.repaired,
    }


@tool("gemini_embedding", [Capability.LLM],
      description="Embed texts for similarity, clustering and deduplication.")
async def gemini_embedding(
    texts: list[str], *, task_type: str = "SEMANTIC_SIMILARITY",
) -> dict[str, Any]:
    if not texts:
        return {"vectors": [], "dim": 0, "model": ""}
    resp = await get_provider().embed(texts, task_type=task_type)
    return {"vectors": resp.vectors, "dim": resp.dim, "model": resp.model}


@tool("llm_status", [Capability.LLM],
      description="Report provider health: key pool, usage, cache, model chains.")
async def llm_status() -> dict[str, Any]:
    return get_provider().status()
