import json
from typing import Any

from llm_gateway.redis_store import RedisStore

PREFIX = "llmgw:cache:exact:"


class ExactCache:
    """Same request -> same saved answer. Stored in Redis with a time-to-live (TTL)."""

    def __init__(self, store: RedisStore, ttl_seconds: int) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds

    async def get(self, fingerprint: str) -> dict[str, Any] | None:
        raw = await self.store.call(lambda r: r.get(PREFIX + fingerprint))
        if raw is None:
            return None
        result: dict[str, Any] = json.loads(raw)
        return result

    async def set(self, fingerprint: str, result: dict[str, Any]) -> None:
        # ex= sets the expiry, so old answers disappear on their own
        await self.store.call(
            lambda r: r.set(PREFIX + fingerprint, json.dumps(result), ex=self.ttl_seconds)
        )
