from collections.abc import Iterator

import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import Config

UPSTREAM_URL = "http://upstream.test/v1/chat/completions"
API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


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


@pytest.fixture
def config() -> Config:
    return Config.model_validate({"upstream": {"url": UPSTREAM_URL, "timeout_seconds": 5}})


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
