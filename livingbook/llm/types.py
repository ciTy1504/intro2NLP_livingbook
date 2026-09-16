"""Provider-neutral request/response types.

Deliberately free of any Gemini vocabulary: adding a second provider means writing an
adapter that speaks these types, with no change to agents, skills or prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ModelRole = Literal["fast", "balanced", "deep", "embedding", "image"]
Role = Literal["system", "user", "model"]


@dataclass(slots=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        # Defined explicitly because slots dataclasses have no __dict__.
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    cached: bool = False
    attempts: int = 1
    raw: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(slots=True)
class StructuredResponse:
    data: Any
    model: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
    attempts: int = 1
    repaired: bool = False


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    model: str
    dim: int
    cached: bool = False


@dataclass(slots=True)
class ImageResponse:
    images: list[bytes]
    model: str
    mime_type: str = "image/png"


@dataclass(slots=True)
class GenerationOptions:
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    system_instruction: str | None = None
    timeout_seconds: float | None = None
    cache: bool = True

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            out["maxOutputTokens"] = self.max_output_tokens
        if self.top_p is not None:
            out["topP"] = self.top_p
        if self.stop_sequences:
            out["stopSequences"] = list(self.stop_sequences)
        return out
