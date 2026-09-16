"""The interface every agent talks to, and the only place keys are handled.

An agent calls ``provider.generate(...)`` with a *role* ("fast" / "balanced" / "deep"),
never a model name and never a key. Model selection, key rotation, rate limiting,
retries, fallback and usage accounting all live below this line.

Adding a provider is implementing this ABC and registering it in ``get_provider``.
No agent, skill or prompt changes.
"""

from __future__ import annotations

import abc
import asyncio
import random
from collections import deque
from typing import Any

from ..config import Config, Secrets, get_config, get_secrets
from .types import (
    EmbeddingResponse,
    GenerationOptions,
    ImageResponse,
    LLMResponse,
    ModelRole,
    StructuredResponse,
)


class LLMProvider(abc.ABC):
    """Provider-neutral LLM interface."""

    name: str = "abstract"

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        role: ModelRole = "balanced",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        ...

    @abc.abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: ModelRole = "balanced",
        options: GenerationOptions | None = None,
    ) -> StructuredResponse:
        ...

    @abc.abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        role: ModelRole = "embedding",
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> EmbeddingResponse:
        ...

    async def generate_image(
        self, prompt: str, *, role: ModelRole = "image"
    ) -> ImageResponse:
        raise NotImplementedError(f"{self.name} does not support image generation")

    async def describe_image(
        self, image_bytes: bytes, prompt: str, *, mime_type: str = "image/png",
        role: ModelRole = "balanced",
    ) -> LLMResponse:
        raise NotImplementedError(f"{self.name} does not support image understanding")

    async def analyse_image_structured(
        self, image_bytes: bytes, prompt: str, schema: dict[str, Any], *,
        mime_type: str = "image/png", role: ModelRole = "balanced",
    ) -> StructuredResponse:
        raise NotImplementedError(f"{self.name} does not support image understanding")

    def status(self) -> dict[str, Any]:
        return {"provider": self.name}

    async def aclose(self) -> None:
        return None


class RateLimiter:
    """Global concurrency cap, minimum spacing, and optional adaptive backoff.

    The cap exists because eight research agents fanning out concurrently can put more
    requests in flight than the shared key pool tolerates.

    Adaptive backoff is implemented but **off by default**, and the reason is worth
    recording. A full book index measured a 14% success rate — six of every seven
    requests were 429s — which looked like the pool being hammered harder than it
    could absorb. It was not: those 429s are cheap (one fast round trip, then a
    different key), and that run still completed 2,173 completions at ~116/min.
    Turning on a global pause made throughput *collapse*, because pausing everything
    to avoid a cheap failure costs far more than the failure does.

    So the mechanism stays, gently tuned, for a provider whose 429s are expensive or
    whose limits are genuinely account-wide. For this pool, measurement says leave it
    off. See `llm.rate_limit.adaptive` in config/config.yaml.
    """

    def __init__(self, max_concurrency: int = 12, min_interval_ms: int = 0,
                 adaptive: bool = False) -> None:
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._min_interval = max(0.0, min_interval_ms / 1000.0)
        self._last = 0.0
        self._lock = asyncio.Lock()
        self._adaptive = adaptive
        self._window: deque[bool] = deque(maxlen=60)   # True = rate limited
        self._penalty = 0.0

    def record(self, rate_limited: bool) -> None:
        if not self._adaptive:
            return
        self._window.append(rate_limited)
        if len(self._window) < 20:
            return
        ratio = sum(self._window) / len(self._window)
        # Gentle: engage only when almost everything is failing, and cap the pause
        # well below the cost of simply retrying on another key.
        if ratio > 0.9:
            self._penalty = min(self._penalty + 0.1, 1.0)
        elif ratio < 0.6:
            self._penalty = max(0.0, self._penalty - 0.2)

    @property
    def penalty_seconds(self) -> float:
        return self._penalty

    async def __aenter__(self) -> "RateLimiter":
        await self._sem.acquire()
        delay = self._penalty
        if self._min_interval or delay:
            # The wait is computed under the lock but slept OUTSIDE it. Sleeping while
            # holding the lock would serialise every concurrent request behind one
            # another — collapsing throughput to one request per penalty interval at
            # precisely the moment the system is already under pressure.
            async with self._lock:
                loop = asyncio.get_running_loop()
                delta = loop.time() - self._last
                wait = max(self._min_interval - delta, 0.0)
                self._last = loop.time() + wait
            if delay:
                wait = max(wait, delay * random.random())
            if wait > 0:
                await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._sem.release()


class RetryPolicy:
    """Exponential backoff with jitter."""

    def __init__(
        self,
        max_attempts: int = 4,
        base_delay: float = 1.5,
        max_delay: float = 45.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def delay_for(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random()
        return min(delay, self.max_delay)


class ModelSelector:
    """Resolves a role to an ordered chain of concrete models.

    A chain rather than a single model because the audit measured that models go
    unavailable independently of keys: pro-tier is 429 on every key, 3.7/3.8-flash are
    503. With one model per role the system would simply stop; with a chain it steps
    sideways and keeps working.
    """

    def __init__(self, chains: dict[str, list[str]]) -> None:
        self._chains = {k: list(v) for k, v in chains.items()}

    def chain(self, role: str) -> list[str]:
        if role not in self._chains:
            raise KeyError(f"Unknown model role {role!r}; known: {sorted(self._chains)}")
        return list(self._chains[role])

    def primary(self, role: str) -> str:
        return self.chain(role)[0]

    def roles(self) -> list[str]:
        return sorted(self._chains)


class UsageTracker:
    """Aggregate token accounting, per model and per role."""

    def __init__(self) -> None:
        self.by_model: dict[str, dict[str, int]] = {}
        self.by_role: dict[str, dict[str, int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_fallbacks = 0
        self.key_rotations = 0

    def record(self, model: str, role: str, tokens_in: int, tokens_out: int) -> None:
        for bucket, key in ((self.by_model, model), (self.by_role, role)):
            slot = bucket.setdefault(key, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
            slot["calls"] += 1
            slot["tokens_in"] += tokens_in
            slot["tokens_out"] += tokens_out

    def snapshot(self) -> dict[str, Any]:
        return {
            "by_model": self.by_model,
            "by_role": self.by_role,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "model_fallbacks": self.model_fallbacks,
            "key_rotations": self.key_rotations,
        }


_provider: LLMProvider | None = None


def get_provider(config: Config | None = None, secrets: Secrets | None = None) -> LLMProvider:
    """Return the configured provider singleton."""
    global _provider
    if _provider is not None:
        return _provider

    cfg = config or get_config()
    sec = secrets or get_secrets()
    name = str(cfg.get("llm.provider", "gemini")).lower()

    if name == "gemini":
        from .gemini import GeminiProvider
        _provider = GeminiProvider(cfg, sec)
    else:
        raise ValueError(
            f"Unknown LLM provider {name!r}. Implement LLMProvider and register it here."
        )
    return _provider


def reset_provider() -> None:
    """Drop the singleton (tests, and CLI commands that reconfigure)."""
    global _provider
    _provider = None
