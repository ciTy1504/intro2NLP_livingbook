"""LLM provider layer.

Agents import from here and nowhere deeper. The rule the whole layer exists to enforce:
an agent asks for a *role*, never a model, and never sees an API key.
"""

from .errors import (
    AllModelsFailed,
    InvalidKey,
    LLMError,
    ModelUnavailable,
    NonRetryable,
    QuotaExhausted,
    RateLimited,
    SchemaValidationError,
    TransientError,
)
from .provider import LLMProvider, get_provider, reset_provider
from .types import (
    EmbeddingResponse,
    GenerationOptions,
    ImageResponse,
    LLMMessage,
    LLMResponse,
    ModelRole,
    StructuredResponse,
    Usage,
)

__all__ = [
    "LLMProvider", "get_provider", "reset_provider",
    "GenerationOptions", "LLMMessage", "LLMResponse", "StructuredResponse",
    "EmbeddingResponse", "ImageResponse", "ModelRole", "Usage",
    "LLMError", "RateLimited", "QuotaExhausted", "InvalidKey",
    "ModelUnavailable", "TransientError", "NonRetryable",
    "AllModelsFailed", "SchemaValidationError",
]
