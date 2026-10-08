"""Translates between OpenAI chat completions and Anthropic's Messages API."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.errors import GatewayError, error_body
from llm_gateway.providers.base import (
    Provider,
    ProviderError,
    connection_error,
    is_retryable_status,
)
from llm_gateway.sse import iter_sse_events, sse_data

ANTHROPIC_VERSION = "2023-06-01"
# Anthropic requires max_tokens; OpenAI doesn't. Used when the client doesn't send one.
DEFAULT_MAX_TOKENS = 16000

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicProvider(Provider):
    def _url(self) -> str:
        return self.config.base_url.rstrip("/") + "/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key or "", "anthropic-version": ANTHROPIC_VERSION}

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.http.post(
                self._url(),
                json=to_anthropic_request(body),
                headers=self._headers(),
                timeout=self.config.timeout_seconds,
            )
        except httpx.TransportError as exc:
            raise connection_error(self.name, exc) from exc
        if response.status_code >= 400:
            raise anthropic_error(response.status_code, response.content)
        return from_anthropic_response(response.json())

    async def open_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        request = self.http.build_request(
            "POST",
            self._url(),
            json=to_anthropic_request(body),
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        try:
            response = await self.http.send(request, stream=True)
        except httpx.TransportError as exc:
            raise connection_error(self.name, exc) from exc
        if response.status_code >= 400:
            content = await response.aread()
            await response.aclose()
            raise anthropic_error(response.status_code, content)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return translate_stream(response, include_usage)


def to_anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("tools"):
        raise GatewayError(
            400, "Tool calls are not supported for Anthropic models yet.", "invalid_request_error"
        )

    # OpenAI puts the system prompt in the message list; Anthropic has a separate field
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in body.get("messages", []):
        role = message.get("role")
        if role in ("system", "developer"):
            system_parts.append(_text_of(message.get("content")))
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": _convert_content(message.get("content"))})
        else:
            raise GatewayError(
                400,
                f"Role '{role}' is not supported for Anthropic models.",
                "invalid_request_error",
            )

    request: dict[str, Any] = {
        "model": body["model"],
        "messages": messages,
        "max_tokens": body.get("max_completion_tokens")
        or body.get("max_tokens")
        or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)
    for key in ("temperature", "top_p"):
        if key in body:
            request[key] = body[key]
    stop = body.get("stop")
    if stop:
        request["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if body.get("stream"):
        request["stream"] = True
    return request


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if part.get("type") == "text")


def _convert_content(content: Any) -> Any:
    """OpenAI content (string or list of parts) -> Anthropic content blocks."""
    if isinstance(content, str) or content is None:
        return content or ""
    blocks = []
    for part in content:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                # "data:image/png;base64,AAAA" -> media type + base64 data
                header, data = url.split(",", 1)
                media_type = header[len("data:") :].split(";")[0]
                source = {"type": "base64", "media_type": media_type, "data": data}
            else:
                source = {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
    return blocks


def from_anthropic_response(data: dict[str, Any]) -> dict[str, Any]:
    # Only text blocks become the answer (thinking blocks are skipped)
    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(data.get("stop_reason") or "", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def translate_stream(response: httpx.Response, include_usage: bool) -> AsyncIterator[bytes]:
    """Anthropic stream events -> OpenAI chat.completion.chunk events."""
    chunk_id, model, created = "", "", int(time.time())
    prompt_tokens = 0

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        return sse_data(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    try:
        async for _event, data in iter_sse_events(response.aiter_bytes()):
            event = json.loads(data)
            kind = event.get("type")
            if kind == "message_start":
                message = event["message"]
                chunk_id, model = message.get("id", ""), message.get("model", "")
                prompt_tokens = message.get("usage", {}).get("input_tokens", 0)
                yield chunk({"role": "assistant", "content": ""})
            elif kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                yield chunk({"content": event["delta"]["text"]})
            elif kind == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason") or ""
                yield chunk({}, FINISH_REASONS.get(stop_reason, "stop"))
                if include_usage:
                    completion_tokens = event.get("usage", {}).get("output_tokens", 0)
                    yield sse_data(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                    )
            elif kind == "message_stop":
                yield sse_data("[DONE]")
            elif kind == "error":
                error = event.get("error", {})
                yield sse_data(error_body(error.get("message", "stream error"), "upstream_error"))
                yield sse_data("[DONE]")
    finally:
        await response.aclose()


def anthropic_error(status_code: int, content: bytes) -> ProviderError:
    """Anthropic's {"type": "error", "error": {...}} -> OpenAI's {"error": {...}}."""
    try:
        error = json.loads(content).get("error", {})
        message, error_type = error.get("message", ""), error.get("type", "upstream_error")
    except ValueError:
        message, error_type = content.decode(errors="replace")[:500], "upstream_error"
    return ProviderError(
        status_code, error_body(message, error_type), is_retryable_status(status_code)
    )
