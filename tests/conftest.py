import os
from collections.abc import Iterator

import pytest
import redis
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Config

OLLAMA_URL = "http://ollama.test/v1/chat/completions"
ANTHROPIC_URL = "http://anthropic.test/v1/messages"
FLAKY_URL = "http://flaky.test/v1/chat/completions"
API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

# Tests use their own Redis database (15) and wipe it, so they never touch real data
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")

# Kept for older tests: the default upstream is the Ollama provider
UPSTREAM_URL = OLLAMA_URL


def chat_response(content: str = "Hello!") -> dict:
    """A minimal OpenAI-style chat completion, like Ollama returns."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "llama3.2:1b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def make_config(**overrides: object) -> Config:
    data: dict = {
        "providers": {
            "ollama": {"type": "openai", "base_url": "http://ollama.test/v1", "timeout_seconds": 5},
            "flaky": {"type": "openai", "base_url": "http://flaky.test/v1"},
            "anthropic": {
                "type": "anthropic",
                "base_url": "http://anthropic.test",
                "api_key_env": "TEST_ANTHROPIC_KEY",
            },
        },
        "models": {
            "llama3.2:1b": {"provider": "ollama", "model": "llama3.2:1b"},
            "smart": {"provider": "ollama", "model": "llama3.2"},
            # Lives on a second provider so tests can make one fail and not the other
            "primary": {"provider": "flaky", "model": "big-model", "fallbacks": ["smart"]},
            "claude": {
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                "drop_params": ["temperature"],
            },
        },
        # No waiting between retries, so tests run instantly
        "retry": {"max_attempts": 3, "initial_backoff_seconds": 0, "max_backoff_seconds": 0},
        "redis": {"url": TEST_REDIS_URL},
    }
    data.update(overrides)
    return Config.model_validate(data)


@pytest.fixture(autouse=True)
def anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-test")


def redis_is_up() -> bool:
    try:
        return bool(redis.Redis.from_url(TEST_REDIS_URL, socket_timeout=0.5).ping())
    except redis.RedisError:
        return False


@pytest.fixture
def clean_redis() -> Iterator[redis.Redis]:
    """For tests that need a real Redis: skips if none is running, wipes the test DB."""
    if not redis_is_up():
        pytest.skip("Redis is not running")
    client = redis.Redis.from_url(TEST_REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()


@pytest.fixture
def config() -> Config:
    return make_config()


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    # Intercepts every httpx request, so tests never touch a real LLM
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def client(config: Config, upstream: respx.MockRouter) -> Iterator[TestClient]:
    app = create_app(config=config, api_keys=[API_KEY])
    # "with" runs the app's startup and shutdown (creates the shared HTTP client)
    with TestClient(app) as test_client:
        yield test_client
