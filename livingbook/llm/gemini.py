"""Gemini adapter.

Sits between the provider interface and the raw Generative Language API, and owns the
two-dimensional failure handling the audit forced on us:

    key dimension    429 / 403 quota  -> cool this key, retry another key, same model
    model dimension  503 / timeout    -> this model is saturated for everyone,
                                          advance the fallback chain

Conflating the two is the classic way a shared pool appears dead: retrying a 503 on
200 different keys burns the whole pool and still fails, while advancing the model
chain on a 429 abandons a model that was fine.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any

import httpx

from ..config import Config, Secrets
from ..obs import get_logger
from .cache import ResponseCache, cache_key
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
    classify_http,
)
from .keypool import KeyPool, NoKeysAvailable, key_fingerprint
from .provider import LLMProvider, ModelSelector, RateLimiter, RetryPolicy, UsageTracker
from .types import (
    EmbeddingResponse,
    GenerationOptions,
    ImageResponse,
    LLMResponse,
    ModelRole,
    StructuredResponse,
    Usage,
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: Config, secrets: Secrets) -> None:
        self.cfg = config
        self.log = get_logger()

        prefixes = config.get("llm.keypool.accepted_prefixes", ["AIzaSy", "AQ.Ab8"])
        keys = secrets.gemini_keys(prefixes)

        self.pool = KeyPool(
            keys,
            cooldown_seconds=float(config.get("llm.keypool.cooldown_seconds", 90)),
            stats_path=config.root / "state" / "keypool_stats.json",
            max_strikes=int(config.get("llm.keypool.max_strikes", 5)),
        )
        self.selector = ModelSelector(config.get("llm.model_roles", {}))
        self.limiter = RateLimiter(
            max_concurrency=int(config.get("llm.rate_limit.max_concurrency", 12)),
            min_interval_ms=int(config.get("llm.rate_limit.min_interval_ms", 0)),
            adaptive=bool(config.get("llm.rate_limit.adaptive", True)),
        )
        self.retry = RetryPolicy(
            max_attempts=int(config.get("llm.retry.max_attempts", 4)),
            base_delay=float(config.get("llm.retry.base_delay_seconds", 1.5)),
            max_delay=float(config.get("llm.retry.max_delay_seconds", 45)),
            jitter=bool(config.get("llm.retry.jitter", True)),
        )
        #: How many times the whole chain is walked before the caller degrades, and
        #: the base pause between walks. A chain that fails in 20s has not really
        #: tried; one that blocks for 10 minutes starves the pipeline behind it.
        self.chain_sweeps = int(config.get("llm.retry.chain_sweeps", 3))
        self.chain_sweep_delay = float(
            config.get("llm.retry.chain_sweep_delay_seconds", 20.0))
        self.usage = UsageTracker()
        self.cache = ResponseCache(
            config.root / "state" / "llm_cache.db",
            ttl_days=int(config.get("llm.cache.ttl_days", 30)),
            enabled=bool(config.get("llm.cache.enabled", True)),
        )
        self.max_key_attempts = int(config.get("llm.keypool.max_key_attempts", 8))
        self.default_timeout = float(config.get("llm.timeouts.request_seconds", 120))
        self.embed_timeout = float(config.get("llm.timeouts.embedding_seconds", 45))
        self.default_temperature = float(config.get("llm.generation.temperature", 0.3))
        self.default_max_tokens = int(config.get("llm.generation.max_output_tokens", 8192))
        self._client: httpx.AsyncClient | None = None

        self.log.info(
            f"gemini provider ready: {len(keys)} keys, roles {self.selector.roles()}"
        )

    # -- transport ---------------------------------------------------------
    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.default_timeout, connect=20.0),
                headers={"User-Agent": "livingbook/0.1", "Content-Type": "application/json"},
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return self._client

    async def aclose(self) -> None:
        self.pool.save_stats()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self.cache.close()

    async def _request(
        self, model: str, endpoint: str, body: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        """One HTTP call against one model with one key, with full key-level handling.

        Raises a classified LLMError. Key-level problems are already reflected in the
        pool before the exception propagates.
        """
        tried: set[str] = set()
        last: LLMError = TransientError("no attempt made", model=model)

        for _ in range(self.max_key_attempts):
            try:
                # Short wait: if every key is cooling, failing fast into the
                # retry/fallback path beats parking this coroutine for minutes.
                key = await self.pool.acquire(exclude=tried, max_wait=45.0)
            except NoKeysAvailable as exc:
                raise QuotaExhausted(str(exc), model=model) from exc
            tried.add(key_fingerprint(key))

            client = await self._http()
            url = f"{API_BASE}/models/{model}:{endpoint}"
            try:
                async with self.limiter:
                    resp = await client.post(
                        url, params={"key": key}, json=body,
                        timeout=httpx.Timeout(timeout, connect=20.0),
                    )
            except (httpx.TimeoutException, httpx.ReadTimeout):
                # A timeout is a model-saturation symptom, not a key problem: the
                # audit saw gemini-3.8-flash time out on 9 of 10 distinct keys.
                raise ModelUnavailable(f"timeout after {timeout:.0f}s", model=model)
            except httpx.HTTPError as exc:
                await self.pool.report_failure(key)
                last = TransientError(f"transport: {type(exc).__name__}: {exc}", model=model)
                continue

            self.limiter.record(resp.status_code in (429, 403))

            if resp.status_code == 200:
                data = resp.json()
                um = data.get("usageMetadata", {}) or {}
                await self.pool.report_success(
                    key,
                    tokens_in=int(um.get("promptTokenCount", 0) or 0),
                    tokens_out=int(um.get("candidatesTokenCount", 0) or 0),
                )
                return data

            err = classify_http(resp.status_code, resp.text, model=model)

            if isinstance(err, InvalidKey):
                await self.pool.report_invalid(
                    key, reason=f"HTTP {resp.status_code} API_KEY_INVALID", immediate=True)
                last = err
                self.usage.key_rotations += 1
                continue
            if isinstance(err, (RateLimited, QuotaExhausted)):
                await self.pool.report_cooldown(
                    key, reason=f"HTTP {resp.status_code}"
                )
                last = err
                self.usage.key_rotations += 1
                continue
            if isinstance(err, ModelUnavailable):
                # Not this key's fault — let the caller advance the model chain.
                raise err
            if isinstance(err, NonRetryable):
                raise err

            await self.pool.report_failure(key)
            last = err

        raise last

    # -- model chain -------------------------------------------------------
    async def _call_with_chain(
        self, role: str, endpoint: str, body_for: Any, *, timeout: float, kind: str,
    ) -> tuple[dict[str, Any], str, int]:
        """Walk the role's model chain, sweeping it again if the whole chain is busy.

        Two nested policies, because "this model is saturated" and "every model is
        saturated" call for different answers.

        Within one sweep, a 503 advances to the next model immediately — retrying a
        model that just said it is overloaded wastes the request. But when the sweep
        ends with every model overloaded, the provider has told us the condition is
        temporary ("Spikes in demand are usually temporary. Please try again"), and
        giving up is wrong: measured, the `deep` chain burned all three models in 22.6
        seconds and dropped a synthesis pass over 12 clusters. So the chain is walked
        again after a real pause.

        Only transient failures earn another sweep. If the chain ended in quota
        exhaustion or invalid keys, nothing will have changed in twenty seconds and
        the caller should degrade now rather than three minutes from now.
        """
        chain = self.selector.chain(role)
        attempts = 0
        errors: list[str] = []

        for sweep in range(1, self.chain_sweeps + 1):
            retryable = False
            for idx, model in enumerate(chain):
                for attempt in range(1, self.retry.max_attempts + 1):
                    attempts += 1
                    try:
                        data = await self._request(
                            model, endpoint, body_for(model), timeout=timeout
                        )
                        if idx > 0 or sweep > 1:
                            self.usage.model_fallbacks += 1
                        return data, model, attempts
                    except ModelUnavailable as exc:
                        errors.append(f"{model}: {exc}")
                        retryable = True
                        self.log.debug(f"{kind}: {model} unavailable, advancing chain")
                        break  # next model — retrying a saturated model wastes time
                    except NonRetryable:
                        # A malformed request is our bug and will fail identically on
                        # every model; surfacing it immediately is the useful behaviour.
                        raise
                    except (RateLimited, QuotaExhausted) as exc:
                        errors.append(f"{model}: {exc}")
                        retryable = retryable or isinstance(exc, RateLimited)
                        if attempt == self.retry.max_attempts:
                            break
                        await asyncio.sleep(self.retry.delay_for(attempt))
                    except (TransientError, LLMError) as exc:
                        errors.append(f"{model}: {exc}")
                        retryable = True
                        if attempt == self.retry.max_attempts:
                            break
                        await asyncio.sleep(self.retry.delay_for(attempt))

            if not retryable or sweep == self.chain_sweeps:
                break
            pause = min(self.chain_sweep_delay * sweep, self.retry.max_delay)
            self.log.warn(
                f"{kind}: whole {role!r} chain busy, sweeping again in {pause:.0f}s "
                f"(sweep {sweep + 1}/{self.chain_sweeps})")
            await asyncio.sleep(pause)

        # One line per distinct failure, not one per attempt: a chain of 3 models at
        # 4 retries each otherwise emits 12 near-identical paragraphs.
        distinct: list[str] = []
        for e in errors:
            if e not in distinct:
                distinct.append(e)
        raise AllModelsFailed(
            f"all models failed for role {role!r} ({kind}): "
            + "; ".join(distinct[:4])
        )

    # -- text generation ---------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        role: ModelRole = "balanced",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        opts = options or GenerationOptions()
        gen_cfg = {
            "temperature": opts.temperature if opts.temperature is not None else self.default_temperature,
            "maxOutputTokens": opts.max_output_tokens or self.default_max_tokens,
        }
        if opts.top_p is not None:
            gen_cfg["topP"] = opts.top_p
        if opts.stop_sequences:
            gen_cfg["stopSequences"] = list(opts.stop_sequences)

        payload = {
            "prompt": prompt,
            "generationConfig": gen_cfg,
            "system": opts.system_instruction,
        }
        chain = self.selector.chain(role)
        ck = cache_key(f"role:{role}:{chain[0]}", "generate", payload)

        if opts.cache:
            hit = self.cache.get(ck)
            if hit is not None:
                self.usage.cache_hits += 1
                return LLMResponse(
                    text=hit["text"], model=hit.get("model", chain[0]),
                    usage=Usage(**hit.get("usage", {})), cached=True,
                )
            self.usage.cache_misses += 1

        def body_for(model: str) -> dict[str, Any]:
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }
            if opts.system_instruction:
                body["systemInstruction"] = {"parts": [{"text": opts.system_instruction}]}
            return body

        data, model, attempts = await self._call_with_chain(
            role, "generateContent", body_for,
            timeout=opts.timeout_seconds or self.default_timeout, kind="generate",
        )
        text, finish = _extract_text(data)
        usage = _extract_usage(data)
        self.usage.record(model, role, usage.prompt_tokens, usage.output_tokens)

        if opts.cache and text.strip():
            self.cache.put(ck, model, "generate", {
                "text": text, "model": model, "usage": usage.as_dict(),
            })

        return LLMResponse(
            text=text, model=model, usage=usage,
            finish_reason=finish, attempts=attempts, raw=data,
        )

    # -- structured output -------------------------------------------------
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: ModelRole = "balanced",
        options: GenerationOptions | None = None,
    ) -> StructuredResponse:
        opts = options or GenerationOptions()
        gen_cfg: dict[str, Any] = {
            "temperature": opts.temperature if opts.temperature is not None else self.default_temperature,
            "maxOutputTokens": opts.max_output_tokens or int(
                self.cfg.get("llm.generation.structured_max_output_tokens", 8192)
            ),
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        payload = {"prompt": prompt, "schema": schema, "generationConfig": gen_cfg,
                   "system": opts.system_instruction}
        chain = self.selector.chain(role)
        ck = cache_key(f"role:{role}:{chain[0]}", "structured", payload)

        if opts.cache:
            hit = self.cache.get(ck)
            # Conform on the way out as well as on the way in: entries written before
            # the check existed still hold the wrong shape, and a cached bad shape is
            # worse than a live one — it fails identically on every retry, which is
            # how one pipeline burned three attempts and went to FAILED.
            cached_data = (_conform_to_schema_shape(hit["data"], schema)
                           if hit is not None else None)
            if cached_data is not None:
                self.usage.cache_hits += 1
                return StructuredResponse(
                    data=cached_data, model=hit.get("model", chain[0]),
                    usage=Usage(**hit.get("usage", {})), cached=True,
                )
            if hit is not None:
                # An unusable entry is not a hit. Generate afresh and let the new
                # answer overwrite it, rather than serving the same broken shape
                # until the ttl expires.
                self.log.debug("structured: cached entry has the wrong shape, regenerating")
            self.usage.cache_misses += 1

        def body_for(model: str) -> dict[str, Any]:
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }
            if opts.system_instruction:
                body["systemInstruction"] = {"parts": [{"text": opts.system_instruction}]}
            return body

        data, model, attempts = await self._call_with_chain(
            role, "generateContent", body_for,
            timeout=opts.timeout_seconds or self.default_timeout, kind="structured",
        )
        text, _ = _extract_text(data)
        usage = _extract_usage(data)
        self.usage.record(model, role, usage.prompt_tokens, usage.output_tokens)

        parsed, repaired = _parse_json(text)
        if parsed is not None:
            parsed = _conform_to_schema_shape(parsed, schema)
        if parsed is None:
            # One repair attempt, feeding the model back its own malformed output.
            repair_prompt = (
                "The following was supposed to be JSON matching the given schema but "
                "could not be parsed. Return ONLY corrected JSON, no commentary.\n\n"
                f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)[:4000]}\n\n"
                f"MALFORMED OUTPUT:\n{text[:12000]}"
            )
            fixed = await self.generate(
                repair_prompt, role="fast",
                options=GenerationOptions(temperature=0.0, cache=False),
            )
            parsed, _ = _parse_json(fixed.text)
            if parsed is not None:
                parsed = _conform_to_schema_shape(parsed, schema)
            repaired = True
            if parsed is None:
                raise SchemaValidationError(
                    f"model {model} returned unparseable JSON ({len(text)} chars)",
                    model=model,
                )

        if opts.cache:
            self.cache.put(ck, model, "structured", {
                "data": parsed, "model": model, "usage": usage.as_dict(),
            })

        return StructuredResponse(
            data=parsed, model=model, usage=usage,
            attempts=attempts, repaired=repaired,
        )

    # -- embeddings --------------------------------------------------------
    async def embed(
        self,
        texts: list[str],
        *,
        role: ModelRole = "embedding",
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(vectors=[], model="", dim=0)

        chain = self.selector.chain(role)
        vectors: list[list[float]] = []
        used_model = chain[0]
        pending: list[tuple[int, str]] = []

        # Serve whatever is cached; only the misses go to the API.
        for i, text in enumerate(texts):
            ck = cache_key(f"role:{role}:{chain[0]}", "embed",
                           {"text": text, "task": task_type})
            hit = self.cache.get(ck)
            if hit is not None:
                self.usage.cache_hits += 1
                vectors.append(hit["vector"])
                used_model = hit.get("model", used_model)
            else:
                self.usage.cache_misses += 1
                vectors.append([])
                pending.append((i, text))

        BATCH = 64
        for start in range(0, len(pending), BATCH):
            chunk = pending[start:start + BATCH]

            def body_for(model: str, _chunk=chunk) -> dict[str, Any]:
                return {
                    "requests": [
                        {
                            "model": f"models/{model}",
                            "content": {"parts": [{"text": t}]},
                            "taskType": task_type,
                        }
                        for _, t in _chunk
                    ]
                }

            data, model, _ = await self._call_with_chain(
                role, "batchEmbedContents", body_for,
                timeout=self.embed_timeout, kind="embed",
            )
            used_model = model
            embeddings = data.get("embeddings", []) or []
            for (idx, text), emb in zip(chunk, embeddings):
                vec = emb.get("values", []) or []
                vectors[idx] = vec
                ck = cache_key(f"role:{role}:{chain[0]}", "embed",
                               {"text": text, "task": task_type})
                self.cache.put(ck, model, "embed", {"vector": vec, "model": model})

        dim = len(vectors[0]) if vectors and vectors[0] else 0
        return EmbeddingResponse(vectors=vectors, model=used_model, dim=dim)

    # -- multimodal --------------------------------------------------------
    async def describe_image(
        self, image_bytes: bytes, prompt: str, *, mime_type: str = "image/png",
        role: ModelRole = "balanced",
    ) -> LLMResponse:
        b64 = base64.b64encode(image_bytes).decode()

        def body_for(model: str) -> dict[str, Any]:
            return {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": mime_type, "data": b64}},
                        {"text": prompt},
                    ],
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
            }

        data, model, attempts = await self._call_with_chain(
            role, "generateContent", body_for,
            timeout=self.default_timeout, kind="describe_image",
        )
        text, finish = _extract_text(data)
        usage = _extract_usage(data)
        self.usage.record(model, role, usage.prompt_tokens, usage.output_tokens)
        return LLMResponse(text=text, model=model, usage=usage,
                           finish_reason=finish, attempts=attempts)

    async def analyse_image_structured(
        self, image_bytes: bytes, prompt: str, schema: dict[str, Any], *,
        mime_type: str = "image/png", role: ModelRole = "balanced",
    ) -> StructuredResponse:
        b64 = base64.b64encode(image_bytes).decode()

        def body_for(model: str) -> dict[str, Any]:
            return {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": mime_type, "data": b64}},
                        {"text": prompt},
                    ],
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                },
            }

        data, model, attempts = await self._call_with_chain(
            role, "generateContent", body_for,
            timeout=self.default_timeout, kind="analyse_image",
        )
        text, _ = _extract_text(data)
        usage = _extract_usage(data)
        self.usage.record(model, role, usage.prompt_tokens, usage.output_tokens)
        parsed, repaired = _parse_json(text)
        if parsed is None:
            raise SchemaValidationError("image analysis returned unparseable JSON", model=model)
        return StructuredResponse(data=parsed, model=model, usage=usage,
                                  attempts=attempts, repaired=repaired)

    async def generate_image(
        self, prompt: str, *, role: ModelRole = "image"
    ) -> ImageResponse:
        def body_for(model: str) -> dict[str, Any]:
            return {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            }

        data, model, _ = await self._call_with_chain(
            role, "generateContent", body_for,
            timeout=self.default_timeout, kind="generate_image",
        )
        images: list[bytes] = []
        mime = "image/png"
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                blob = part.get("inlineData") or part.get("inline_data")
                if blob and blob.get("data"):
                    images.append(base64.b64decode(blob["data"]))
                    mime = blob.get("mimeType") or blob.get("mime_type") or mime
        if not images:
            raise ModelUnavailable("image model returned no image data", model=model)
        return ImageResponse(images=images, model=model, mime_type=mime)

    # -- introspection -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "keypool": self.pool.status(),
            "usage": self.usage.snapshot(),
            "cache": self.cache.stats(),
            "roles": {r: self.selector.chain(r) for r in self.selector.roles()},
        }

    async def list_models(self) -> list[str]:
        key = await self.pool.acquire()
        client = await self._http()
        resp = await client.get(
            f"{API_BASE}/models", params={"key": key, "pageSize": 200}
        )
        resp.raise_for_status()
        return sorted(
            m["name"].replace("models/", "") for m in resp.json().get("models", [])
        )


# -- response parsing ------------------------------------------------------


def _extract_text(data: dict[str, Any]) -> tuple[str, str | None]:
    candidates = data.get("candidates") or []
    if not candidates:
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise NonRetryable(f"prompt blocked: {block}")
        return "", None
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return text, cand.get("finishReason")


def _extract_usage(data: dict[str, Any]) -> Usage:
    um = data.get("usageMetadata") or {}
    return Usage(
        prompt_tokens=int(um.get("promptTokenCount", 0) or 0),
        output_tokens=int(um.get("candidatesTokenCount", 0) or 0),
        total_tokens=int(um.get("totalTokenCount", 0) or 0),
    )


def _parse_json(text: str) -> tuple[Any | None, bool]:
    """Parse model output as JSON, tolerating fences and surrounding prose.

    ``responseSchema`` usually makes this unnecessary, but a fallback model in the
    chain may not honour it, and a hard failure there would abort an otherwise fine
    pipeline step.
    """
    if not text or not text.strip():
        return None, False

    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    fence = _JSON_FENCE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1)), True
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1]), True
            except json.JSONDecodeError:
                continue

    return None, False


def _conform_to_schema_shape(parsed: Any, schema: dict[str, Any]) -> Any:
    """Make the top-level shape match what the schema asked for.

    A schema declaring OBJECT is sometimes answered with a one-element array holding
    that object. Every caller then does data.get(...) on a list and dies with
    "'list' object has no attribute 'get'" — which is what took the citation
    verifier down three attempts in a row and failed the pipeline outright.

    Unwrapping here rather than at each call site because the shape is the provider's
    contract to keep, and there are dozens of callers who would otherwise each need
    the same defensive check.
    """
    want = str(schema.get("type", "")).upper()
    if want == "OBJECT" and isinstance(parsed, list):
        objects = [x for x in parsed if isinstance(x, dict)]
        if len(objects) == 1:
            return objects[0]
        if objects:
            # Several objects where one was asked for: merge shallowly, first wins,
            # which preserves the primary answer and keeps any fields it omitted.
            merged: dict[str, Any] = {}
            for obj in reversed(objects):
                merged.update(obj)
            return merged
        # A list holding no object at all — measured: a bare list of caveat strings
        # where the whole verification object was asked for. Nothing here can be
        # reshaped into the answer, so report it as unusable and let the caller's
        # repair pass ask again, rather than handing back something that crashes
        # every consumer with "'list' object has no attribute 'get'".
        return None
    if want == "OBJECT" and parsed is not None and not isinstance(parsed, dict):
        return None
    if want == "ARRAY" and isinstance(parsed, dict):
        # The mirror image: a single item returned bare instead of in a list.
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return [parsed]
    return parsed
