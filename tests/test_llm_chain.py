"""The model chain under provider-wide congestion.

The distinction these tests pin down: a saturated *model* should be stepped over
immediately, but a saturated *chain* is a temporary condition the provider tells us to
retry. Conflating them cost a synthesis pass over 12 clusters.
"""

import asyncio

import pytest

from livingbook.llm.errors import (
    AllModelsFailed,
    LLMError,
    ModelUnavailable,
    NonRetryable,
    QuotaExhausted,
)


class _Chain:
    """The parts of GeminiProvider._call_with_chain relies on, and nothing else."""

    def __init__(self, models, responses, *, sweeps=3, delay=0.0, max_attempts=1):
        from livingbook.llm.provider import ModelSelector, RetryPolicy
        from livingbook.obs import get_logger

        self.selector = ModelSelector({"deep": models})
        self.retry = RetryPolicy(max_attempts=max_attempts, base_delay=0.0,
                                 max_delay=0.0, jitter=False)
        self.chain_sweeps = sweeps
        self.chain_sweep_delay = delay
        self.log = get_logger()
        self.usage = type("U", (), {"model_fallbacks": 0})()
        self.responses = responses
        self.calls = []

    async def _request(self, model, endpoint, body, *, timeout):
        self.calls.append(model)
        outcome = self.responses(model, len(self.calls))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def run(self):
        from livingbook.llm.gemini import GeminiProvider
        return await GeminiProvider._call_with_chain(
            self, "deep", "generateContent", lambda m: {}, timeout=1.0, kind="generate")


def test_a_busy_chain_is_swept_again_rather_than_abandoned():
    """Every model 503s on the first pass, then the first model recovers.

    This is the measured failure: three models reported "high demand ... try again"
    inside 22 seconds and the caller degraded. One more sweep gets the answer.
    """
    models = ["m-a", "m-b", "m-c"]

    def responses(model, n):
        if n <= 3:
            return ModelUnavailable("model overloaded: UNAVAILABLE", model=model)
        return {"ok": True}

    chain = _Chain(models, responses)
    data, model, attempts = asyncio.run(chain.run())

    assert data == {"ok": True}
    assert model == "m-a", "the second sweep should start from the head of the chain"
    assert chain.calls == ["m-a", "m-b", "m-c", "m-a"]


def test_quota_exhaustion_does_not_earn_another_sweep():
    """Nothing changes in twenty seconds, so the caller should degrade now."""
    models = ["m-a", "m-b"]
    chain = _Chain(models,
                   lambda model, n: QuotaExhausted("RESOURCE_EXHAUSTED", model=model))

    with pytest.raises(AllModelsFailed):
        asyncio.run(chain.run())

    assert chain.calls == ["m-a", "m-b"], "quota failures must not be swept"


def test_a_saturated_model_is_stepped_over_without_retrying_it():
    """Within a sweep, a 503 advances immediately — retrying it wastes the request."""
    models = ["m-a", "m-b"]

    def responses(model, n):
        if model == "m-a":
            return ModelUnavailable("model overloaded", model=model)
        return {"ok": True}

    chain = _Chain(models, responses, max_attempts=4)
    data, model, _ = asyncio.run(chain.run())

    assert model == "m-b"
    assert chain.calls.count("m-a") == 1, "a 503 must not be retried on the same model"


def test_a_malformed_request_fails_immediately_on_the_first_model():
    """Our bug, not the provider's; it will fail identically everywhere."""
    chain = _Chain(["m-a", "m-b"],
                   lambda model, n: NonRetryable("unsupported parameter", model=model))

    with pytest.raises(NonRetryable):
        asyncio.run(chain.run())

    assert chain.calls == ["m-a"]


def test_sweeps_are_bounded():
    """A chain that never recovers must still hand back to the caller."""
    chain = _Chain(["m-a", "m-b"],
                   lambda model, n: ModelUnavailable("overloaded", model=model),
                   sweeps=2)

    with pytest.raises(AllModelsFailed):
        asyncio.run(chain.run())

    assert chain.calls == ["m-a", "m-b", "m-a", "m-b"]
