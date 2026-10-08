"""The request pipeline: everything between "request arrived" and "response sent".

app.py handles HTTP; this module decides what happens to a chat request:
cache lookup -> provider call (with retries and fallback) -> cache store.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.cache import is_cacheable_request, is_cacheable_response, request_fingerprint
from llm_gateway.cache.exact import ExactCache
from llm_gateway.config import Config
from llm_gateway.router import Router


@dataclass
class ChatResult:
    """Either a full JSON answer or a stream, plus headers describing what happened."""

    body: dict[str, Any] | None = None
    stream: AsyncIterator[bytes] | None = None
    headers: dict[str, str] = field(default_factory=dict)


class Gateway:
    def __init__(self, config: Config, router: Router, exact_cache: ExactCache | None) -> None:
        self.config = config
        self.router = router
        self.exact_cache = exact_cache

    async def chat(self, body: dict[str, Any], key_id: str, skip_cache: bool = False) -> ChatResult:
        if body.get("stream"):
            # Streams aren't cached: the answer is sent piece by piece as it's generated
            stream, target = await self.router.open_stream(body)
            return ChatResult(stream=stream, headers={"x-gateway-model": target.name})

        target = self.router.resolve(body["model"])
        cache_status = "skip"
        fingerprint = None
        if (
            self.exact_cache
            and not skip_cache
            and is_cacheable_request(body, self.config.cache.exact.max_temperature)
        ):
            scope = "shared" if self.config.cache.shared_across_keys else key_id
            model_id = f"{target.model.provider}/{target.model.model}"
            fingerprint = request_fingerprint(body, model_id, scope)
            cached = await self.exact_cache.get(fingerprint)
            if cached is not None:
                return ChatResult(
                    body=cached, headers={"x-gateway-model": target.name, "x-gateway-cache": "hit"}
                )
            cache_status = "miss"

        result, served_by = await self.router.chat(body)

        # Only cache answers from the model that was asked for (not a fallback's answer)
        if (
            fingerprint
            and self.exact_cache
            and served_by.name == target.name
            and is_cacheable_response(result)
        ):
            await self.exact_cache.set(fingerprint, result)

        return ChatResult(
            body=result,
            headers={"x-gateway-model": served_by.name, "x-gateway-cache": cache_status},
        )
