import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import error_body


class ProviderError(Exception):
    """The provider failed. Carries an OpenAI-style error body to send back to the client."""

    def __init__(self, status_code: int, body: dict[str, Any], retryable: bool) -> None:
        super().__init__(body.get("error", {}).get("message", "provider error"))
        self.status_code = status_code
        self.body = body
        # Worth trying again (or trying another model)? True for timeouts, 429 and 5xx
        self.retryable = retryable


def is_retryable_status(status_code: int) -> bool:
    return status_code in (408, 409, 429) or status_code >= 500


def connection_error(provider: str, exc: httpx.TransportError) -> ProviderError:
    """Turn a network failure (refused, timed out, ...) into a 502/504 for the client."""
    if isinstance(exc, httpx.TimeoutException):
        status, message = 504, f"Provider '{provider}' timed out"
    else:
        status, message = 502, f"Could not reach provider '{provider}'"
    return ProviderError(
        status, error_body(message, "upstream_error", "provider_unavailable"), True
    )


class Provider(ABC):
    """Talks to one LLM API. Input and output are always in OpenAI's format."""

    def __init__(self, name: str, config: ProviderConfig, http: httpx.AsyncClient) -> None:
        self.name = name
        self.config = config
        self.http = http
        self.api_key = None
        if config.api_key_env:
            # Read the secret from the environment (.env), never from config.yaml
            self.api_key = _require_env(config.api_key_env, name)

    @abstractmethod
    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send a chat request and return the full OpenAI-style response."""

    @abstractmethod
    async def open_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Start a streaming request. Raises ProviderError before any bytes if it can't start."""


def _require_env(var: str, provider: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"Provider '{provider}' needs {var} to be set in .env")
    return value
