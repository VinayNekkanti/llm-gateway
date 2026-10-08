from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import redis
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.auth import key_id
from llm_gateway.config import RateLimitConfig
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.redis_store import RedisStore
from tests.conftest import API_KEY, AUTH, OLLAMA_URL, TEST_REDIS_URL, chat_response, make_config


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


MakeLimiter = Callable[..., RateLimiter]


@pytest.fixture
async def make_limiter() -> AsyncIterator[MakeLimiter]:
    """Builds rate limiters on the test Redis and closes their connections afterwards."""
    stores: list[RedisStore] = []

    def build(
        clock: FakeClock, limit: int = 3, url: str = TEST_REDIS_URL, **extra: Any
    ) -> RateLimiter:
        store = RedisStore(url, 0.5)
        stores.append(store)
        config = RateLimitConfig(requests_per_minute=limit, window_seconds=60, **extra)
        return RateLimiter(store, config, clock)

    yield build
    for store in stores:
        await store.close()


async def test_allows_up_to_limit_then_rejects(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    clock = FakeClock(600.0)  # exactly at the start of a window
    rl = make_limiter(clock)

    results = [(await rl.check("k")).allowed for _ in range(4)]

    assert results == [True, True, True, False]


async def test_rejected_requests_do_not_use_quota(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    clock = FakeClock(600.0)
    rl = make_limiter(clock, limit=1)
    await rl.check("k")
    for _ in range(5):
        await rl.check("k")  # all rejected

    clock.now = 600.0 + 120  # two windows later, everything has expired from the window
    assert (await rl.check("k")).allowed


async def test_previous_window_still_counts(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    # 3 requests at the end of one minute, then a new minute starts.
    # A plain per-minute counter would allow 3 more immediately; the sliding window doesn't.
    clock = FakeClock(659.0)
    rl = make_limiter(clock)
    for _ in range(3):
        await rl.check("k")

    clock.now = 661.0  # 1s into the next window: previous window still ~98% in range
    decision = await rl.check("k")

    assert not decision.allowed
    assert decision.retry_after_seconds >= 1


async def test_previous_window_fades_out(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    clock = FakeClock(659.0)
    rl = make_limiter(clock)
    for _ in range(3):
        await rl.check("k")

    clock.now = 660.0 + 40  # 40s into the next window: only 1/3 of the old window overlaps
    assert (await rl.check("k")).allowed


async def test_retry_after_when_current_window_full(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    clock = FakeClock(610.0)  # 10s into the window
    rl = make_limiter(clock, limit=2)
    await rl.check("k")
    await rl.check("k")

    decision = await rl.check("k")

    assert not decision.allowed
    assert decision.retry_after_seconds == 50  # until this window ends


async def test_keys_are_limited_separately(
    clean_redis: redis.Redis, make_limiter: MakeLimiter
) -> None:
    rl = make_limiter(FakeClock(600.0), limit=1)
    assert (await rl.check("a")).allowed
    assert (await rl.check("b")).allowed
    assert not (await rl.check("a")).allowed


async def test_per_key_override(clean_redis: redis.Redis, make_limiter: MakeLimiter) -> None:
    rl = make_limiter(FakeClock(600.0), limit=1, per_key={"vip": 3})
    assert [(await rl.check("vip")).allowed for _ in range(4)] == [True, True, True, False]


async def test_fails_open_when_redis_is_down(make_limiter: MakeLimiter) -> None:
    rl = make_limiter(FakeClock(600.0), limit=1, url="redis://localhost:1/0")
    assert all([(await rl.check("k")).allowed for _ in range(3)])


@pytest.fixture
def limited_client(upstream: respx.MockRouter, clean_redis: redis.Redis) -> TestClient:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    config = make_config(rate_limit={"requests_per_minute": 2})
    return TestClient(create_app(config=config, api_keys=[API_KEY]))


def test_over_limit_gets_429_with_retry_after(limited_client: TestClient) -> None:
    body = {"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
    with limited_client as client:
        codes = [client.post("/v1/chat/completions", json=body, headers=AUTH) for _ in range(3)]

    assert [r.status_code for r in codes] == [200, 200, 429]
    rejected = codes[2]
    assert rejected.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(rejected.headers["Retry-After"]) >= 1
    assert rejected.headers["X-RateLimit-Limit"] == "2"


def test_rate_limit_checks_key_first(limited_client: TestClient) -> None:
    body = {"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
    with limited_client as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 401


def test_health_is_not_rate_limited(limited_client: TestClient) -> None:
    with limited_client as client:
        assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_key_id_used_for_overrides() -> None:
    # Overrides in config.yaml use the key id, so the raw key never goes in a config file
    assert key_id(API_KEY).startswith("key_")
