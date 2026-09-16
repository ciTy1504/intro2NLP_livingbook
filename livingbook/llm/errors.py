"""LLM error taxonomy.

The classification matters because each class has a different correct response, and
getting this wrong is the difference between a working pool and a stalled one:

  RateLimited / QuotaExhausted -> the KEY is spent. Cool it down, retry on another key,
                                  same model.
  ModelUnavailable             -> the MODEL is saturated (503/timeout). Changing keys
                                  will not help. Advance the model fallback chain.
  InvalidKey                   -> the key is structurally dead. Disable it permanently.
  TransientError               -> retry with backoff.
  NonRetryable                 -> a bad request. Fail the step; retrying is waste.

Measured on 2026-09-16: pro-tier models return 429 on every key while
gemini-3.7/3.8-flash return 503 on all of them. Both had to be survivable, and they
need opposite responses, which is why this distinction exists at all.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base for every provider-layer failure."""

    def __init__(self, message: str, *, model: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.model = model
        self.status = status


class RateLimited(LLMError):
    """HTTP 429 — this key is rate limited right now."""


class QuotaExhausted(LLMError):
    """HTTP 403 with a quota reason — this key has no remaining allowance."""


class InvalidKey(LLMError):
    """The credential itself is rejected; disable it permanently."""


class ModelUnavailable(LLMError):
    """HTTP 503 / 404 / timeout — the model, not the key, is the problem."""


class TransientError(LLMError):
    """Network blip or 5xx; worth retrying with backoff."""


class NonRetryable(LLMError):
    """Malformed request, unsupported parameter, safety block. Do not retry."""


class AllModelsFailed(LLMError):
    """Every model in the role's fallback chain was exhausted."""


class SchemaValidationError(LLMError):
    """The model returned JSON that does not satisfy the requested schema."""


def classify_http(status: int, body: str, model: str | None = None) -> LLMError:
    """Map an HTTP status + response body onto the taxonomy above."""
    lowered = (body or "").lower()

    if status == 400:
        if "api_key_invalid" in lowered or "api key not valid" in lowered:
            return InvalidKey(f"key rejected: {body[:200]}", model=model, status=status)
        return NonRetryable(f"bad request: {body[:300]}", model=model, status=status)

    if status == 429:
        return RateLimited(f"rate limited: {body[:200]}", model=model, status=status)

    if status == 403:
        # A 403 is NOT treated as a dead key. Measured: random keys return an
        # occasional 403 on a specific model while working fine on the next call, so
        # disabling on a single 403 silently shrinks the pool for no reason. Only an
        # explicit API_KEY_INVALID (a 400) is structural; everything else cools down
        # and the pool's strike counter decides if a key is genuinely dead.
        if "api key not valid" in lowered or "api_key_invalid" in lowered:
            return InvalidKey(f"key rejected: {body[:200]}", model=model, status=status)
        if "quota" in lowered or "exhaust" in lowered or "billing" in lowered:
            return QuotaExhausted(f"quota exhausted: {body[:200]}", model=model, status=status)
        return QuotaExhausted(f"forbidden: {body[:200]}", model=model, status=status)

    if status in (404, 501):
        return ModelUnavailable(f"model unavailable: {body[:200]}", model=model, status=status)

    if status == 503:
        return ModelUnavailable(f"model overloaded: {body[:200]}", model=model, status=status)

    if 500 <= status < 600:
        return TransientError(f"server error {status}: {body[:200]}", model=model, status=status)

    return TransientError(f"unexpected status {status}: {body[:200]}", model=model, status=status)
