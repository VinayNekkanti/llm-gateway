import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RedisStore:
    """Redis that fails soft: if Redis is down, calls return None and the gateway keeps serving.

    After an error we skip Redis for `cooldown_seconds`, so every request doesn't pay a
    connection timeout while Redis is down (a simple circuit breaker).
    """

    def __init__(self, url: str, timeout_seconds: float, cooldown_seconds: float = 5.0) -> None:
        self.client: Redis = Redis.from_url(
            url,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            decode_responses=False,
        )
        self.cooldown_seconds = cooldown_seconds
        self._down_until = 0.0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._down_until

    async def call(self, operation: Callable[[Redis], Awaitable[T]]) -> T | None:
        if not self.available:
            return None
        try:
            return await operation(self.client)
        except (RedisError, OSError) as exc:
            logger.warning("Redis unavailable, continuing without it: %s", exc)
            self._down_until = time.monotonic() + self.cooldown_seconds
            return None

    async def close(self) -> None:
        await self.client.aclose()
