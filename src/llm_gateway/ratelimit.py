"""Per-API-key rate limiting with a sliding window counter, stored in Redis.

A plain "N requests per clock minute" counter lets a client send 2N requests in a burst
around the minute boundary. The sliding window counter fixes that cheaply: it blends the
previous minute's count (weighted by how much of it still overlaps the last 60 seconds)
with the current minute's count. Only two small counters per key, no per-request log.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Request
from redis.asyncio import Redis

from llm_gateway.auth import require_api_key
from llm_gateway.config import RateLimitConfig
from llm_gateway.errors import GatewayError
from llm_gateway.observability import RATE_LIMITED
from llm_gateway.redis_store import RedisStore

PREFIX = "llmgw:ratelimit:"


@dataclass
class Decision:
    allowed: bool
    limit: int
    retry_after_seconds: int = 0


class RateLimiter:
    def __init__(
        self,
        store: RedisStore,
        config: RateLimitConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.config = config
        self.clock = clock

    def limit_for(self, key_id: str) -> int:
        return self.config.per_key.get(key_id, self.config.requests_per_minute)

    async def check(self, key_id: str) -> Decision:
        limit = self.limit_for(key_id)
        window = self.config.window_seconds
        now = self.clock()
        window_index = int(now // window)
        elapsed = now - window_index * window
        current_key = f"{PREFIX}{key_id}:{window_index}"
        previous_key = f"{PREFIX}{key_id}:{window_index - 1}"

        async def count(redis: Redis) -> list[Any]:
            # MULTI/EXEC: these commands run together, so concurrent requests can't interleave
            pipe = redis.pipeline(transaction=True)
            pipe.get(previous_key)
            pipe.incr(current_key)
            pipe.expire(current_key, window * 2)
            result: list[Any] = await pipe.execute()
            return result

        result = await self.store.call(count)
        if result is None:
            # Redis is down: fail open (allow). Blocking all traffic would turn a cache
            # outage into a full outage.
            return Decision(True, limit)

        previous = int(result[0] or 0)
        current = int(result[1])
        overlap = (window - elapsed) / window  # share of the previous window still in range
        estimated = previous * overlap + current
        if estimated <= limit:
            return Decision(True, limit)

        # Rejected requests shouldn't use up quota, or a client retrying in a loop never recovers
        await self.store.call(lambda r: r.decr(current_key))
        return Decision(False, limit, self._retry_after(previous, current - 1, elapsed, limit))

    def _retry_after(self, previous: int, current: int, elapsed: float, limit: int) -> int:
        window = self.config.window_seconds
        until_next_window = window - elapsed
        if current + 1 > limit or previous == 0:
            # This window alone is full: wait for it to end
            return max(1, math.ceil(until_next_window))
        # Otherwise wait until enough of the previous window has slid out of range:
        # previous * (window - elapsed - t) / window + current + 1 <= limit
        wait = until_next_window - (limit - current - 1) * window / previous
        return max(1, math.ceil(wait))


async def enforce_rate_limit(request: Request, key: str = Depends(require_api_key)) -> str:
    """FastAPI dependency: authenticate, then reject with 429 if this key is over its limit."""
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is None:
        return key
    decision = await limiter.check(key)
    if not decision.allowed:
        RATE_LIMITED.inc()
        raise GatewayError(
            429,
            f"Rate limit reached ({decision.limit} requests per "
            f"{limiter.config.window_seconds}s). Try again in {decision.retry_after_seconds}s.",
            "rate_limit_error",
            "rate_limit_exceeded",
            headers={
                "Retry-After": str(decision.retry_after_seconds),
                "X-RateLimit-Limit": str(decision.limit),
            },
        )
    return key
