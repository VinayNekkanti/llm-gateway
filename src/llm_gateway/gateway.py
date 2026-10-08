"""The request pipeline: everything between "request arrived" and "response sent".

app.py handles HTTP; this module decides what happens to a chat request:
exact cache -> semantic cache -> provider call (with retries and fallback) -> cache store.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.cache import is_cacheable_request, is_cacheable_response, request_fingerprint
from llm_gateway.cache.exact import ExactCache
from llm_gateway.cache.semantic import SemanticCache, context_hash, split_request
from llm_gateway.config import Config
from llm_gateway.observability import CACHE_LOOKUPS, COST_USD, TOKENS
from llm_gateway.router import Router, Target
from llm_gateway.usage import UsageTracker, cost_micro_usd, token_counts, track_stream


@dataclass
class ChatResult:
    """Either a full JSON answer or a stream, plus headers describing what happened."""

    body: dict[str, Any] | None = None
    stream: AsyncIterator[bytes] | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticKey:
    question: str
    vector: list[float]
    ctx: str


class Gateway:
    def __init__(
        self,
        config: Config,
        router: Router,
        exact_cache: ExactCache | None,
        semantic_cache: SemanticCache | None = None,
        usage: UsageTracker | None = None,
    ) -> None:
        self.config = config
        self.router = router
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.usage = usage

    async def chat(self, body: dict[str, Any], key_id: str, skip_cache: bool = False) -> ChatResult:
        if body.get("stream"):
            return await self._stream(body, key_id)
        result = await self._chat(body, key_id, skip_cache)
        CACHE_LOOKUPS.labels(result.headers.get("x-gateway-cache", "skip")).inc()
        return result

    async def _chat(self, body: dict[str, Any], key_id: str, skip_cache: bool) -> ChatResult:
        target = self.router.resolve(body["model"])
        headers = {"x-gateway-model": target.name, "x-gateway-cache": "skip"}
        use_cache = (
            (self.exact_cache or self.semantic_cache)
            and not skip_cache
            and is_cacheable_request(body, self.config.cache.max_temperature)
        )
        fingerprint: str | None = None
        semantic_key: SemanticKey | None = None

        if use_cache:
            scope = "shared" if self.config.cache.shared_across_keys else key_id
            model_id = f"{target.model.provider}/{target.model.model}"
            headers["x-gateway-cache"] = "miss"

            # 1. Exact match: cheapest check, so it goes first
            if self.exact_cache:
                fingerprint = request_fingerprint(body, model_id, scope)
                cached = await self.exact_cache.get(fingerprint)
                if cached is not None:
                    await self._record(key_id, target, cached.get("usage"), cache_hit=True)
                    return ChatResult(body=cached, headers={**headers, "x-gateway-cache": "hit"})

            # 2. Same meaning: embed the question and search for a close enough earlier one
            semantic_key = await self._semantic_key(body, model_id, scope)
            if self.semantic_cache and semantic_key:
                found = await self.semantic_cache.lookup(
                    semantic_key.question, semantic_key.vector, semantic_key.ctx
                )
                if found is not None:
                    answer, similarity = found
                    await self._record(key_id, target, answer.get("usage"), cache_hit=True)
                    return ChatResult(
                        body=answer,
                        headers={
                            **headers,
                            "x-gateway-cache": "semantic-hit",
                            "x-gateway-similarity": f"{similarity:.3f}",
                        },
                    )

        result, served_by = await self.router.chat(body)
        headers["x-gateway-model"] = served_by.name
        cost = await self._record(key_id, served_by, result.get("usage"))
        headers["x-gateway-cost-usd"] = f"{cost / 1_000_000:.6f}"

        # Only cache good answers from the model that was asked for (not a fallback's answer)
        if served_by.name == target.name and is_cacheable_response(result):
            if self.exact_cache and fingerprint:
                await self.exact_cache.set(fingerprint, result)
            if self.semantic_cache and semantic_key:
                await self.semantic_cache.store_answer(
                    semantic_key.question, semantic_key.vector, semantic_key.ctx, result
                )

        return ChatResult(body=result, headers=headers)

    async def _stream(self, body: dict[str, Any], key_id: str) -> ChatResult:
        # Streams aren't cached: the answer is sent piece by piece as it's generated.
        # Always ask the provider for token usage (sent in the last chunk) so we can bill it.
        client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        upstream_body = {
            **body,
            "stream_options": {**(body.get("stream_options") or {}), "include_usage": True},
        }
        chunks, target = await self.router.open_stream(upstream_body)

        async def on_usage(usage: dict[str, Any] | None) -> None:
            await self._record(key_id, target, usage)

        return ChatResult(
            stream=track_stream(chunks, client_wants_usage, on_usage),
            headers={"x-gateway-model": target.name},
        )

    async def _record(
        self,
        key_id: str,
        target: Target,
        usage: dict[str, Any] | None,
        cache_hit: bool = False,
    ) -> int:
        if not cache_hit:
            prompt_tokens, completion_tokens = token_counts(usage)
            TOKENS.labels(target.name, "prompt").inc(prompt_tokens)
            TOKENS.labels(target.name, "completion").inc(completion_tokens)
            cost = cost_micro_usd(target.model, prompt_tokens, completion_tokens)
            COST_USD.labels(target.name).inc(cost / 1_000_000)
        if self.usage is None:
            return 0
        return await self.usage.record(key_id, target.name, target.model, usage, cache_hit)

    async def _semantic_key(
        self, body: dict[str, Any], model_id: str, scope: str
    ) -> SemanticKey | None:
        # Skip the embedding work entirely if Redis is down: nothing to search or store
        if not self.semantic_cache or not self.semantic_cache.store.available:
            return None
        parts = split_request(body)
        if parts is None:
            return None
        question, earlier, params = parts
        vector = await self.semantic_cache.embed(question)
        return SemanticKey(question, vector, context_hash(earlier, params, model_id, scope))
