from collections.abc import Iterator

import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Config

OLLAMA_URL = "http://ollama.test/v1/chat/completions"
ANTHROPIC_URL = "http://anthropic.test/v1/messages"
API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

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
            "anthropic": {
                "type": "anthropic",
                "base_url": "http://anthropic.test",
                "api_key_env": "TEST_ANTHROPIC_KEY",
            },
        },
        "models": {
            "llama3.2:1b": {"provider": "ollama", "model": "llama3.2:1b"},
            "smart": {"provider": "ollama", "model": "llama3.2"},
            "claude": {
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                "drop_params": ["temperature"],
            },
        },
    }
    data.update(overrides)
    return Config.model_validate(data)


@pytest.fixture(autouse=True)
def anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-test")


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
