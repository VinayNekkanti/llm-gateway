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
from llm_gateway.router import Router


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
    ) -> None:
        self.config = config
        self.router = router
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache

    async def chat(self, body: dict[str, Any], key_id: str, skip_cache: bool = False) -> ChatResult:
        if body.get("stream"):
            # Streams aren't cached: the answer is sent piece by piece as it's generated
            stream, target = await self.router.open_stream(body)
            return ChatResult(stream=stream, headers={"x-gateway-model": target.name})

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
                    return ChatResult(body=cached, headers={**headers, "x-gateway-cache": "hit"})

            # 2. Same meaning: embed the question and search for a close enough earlier one
            semantic_key = await self._semantic_key(body, model_id, scope)
            if self.semantic_cache and semantic_key:
                found = await self.semantic_cache.lookup(
                    semantic_key.question, semantic_key.vector, semantic_key.ctx
                )
                if found is not None:
                    answer, similarity = found
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

        # Only cache good answers from the model that was asked for (not a fallback's answer)
        if served_by.name == target.name and is_cacheable_response(result):
            if self.exact_cache and fingerprint:
                await self.exact_cache.set(fingerprint, result)
            if self.semantic_cache and semantic_key:
                await self.semantic_cache.store_answer(
                    semantic_key.question, semantic_key.vector, semantic_key.ctx, result
                )

        return ChatResult(body=result, headers=headers)

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
