import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.errors import error_body
from llm_gateway.providers.base import (
    Provider,
    ProviderError,
    connection_error,
    is_retryable_status,
)


class OpenAICompatibleProvider(Provider):
    """Any API that already speaks OpenAI's format (Ollama, OpenAI, vLLM): no translation needed."""

    def _url(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.http.post(
                self._url(),
                json=body,
                headers=self._headers(),
                timeout=self.config.timeout_seconds,
            )
        except httpx.TransportError as exc:
            raise connection_error(self.name, exc) from exc
        if response.status_code >= 400:
            raise _status_error(response.status_code, response.content)
        result: dict[str, Any] = response.json()
        return result

    async def open_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        request = self.http.build_request(
            "POST",
            self._url(),
            json=body,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        try:
            response = await self.http.send(request, stream=True)
        except httpx.TransportError as exc:
            raise connection_error(self.name, exc) from exc
        # Check the status before streaming, so errors become a normal error response
        if response.status_code >= 400:
            content = await response.aread()
            await response.aclose()
            raise _status_error(response.status_code, content)
        return _pass_through(response)


async def _pass_through(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def _status_error(status_code: int, content: bytes) -> ProviderError:
    try:
        body = json.loads(content)
        if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
            raise ValueError
    except ValueError:
        # Not an OpenAI-style error body: wrap whatever the provider said
        body = error_body(content.decode(errors="replace")[:500], "upstream_error")
    return ProviderError(status_code, body, is_retryable_status(status_code))
