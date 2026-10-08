import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.errors import GatewayError
from llm_gateway.providers.anthropic import (
    from_anthropic_response,
    to_anthropic_request,
)
from tests.conftest import ANTHROPIC_URL, AUTH


def test_request_moves_system_prompt_and_sets_max_tokens() -> None:
    request = to_anthropic_request(
        {
            "model": "claude-opus-5-5",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            "stop": "END",
        }
    )
    assert request["system"] == "Be brief."
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["max_tokens"] > 0
    assert request["stop_sequences"] == ["END"]


def test_request_uses_client_max_tokens() -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50}
    assert to_anthropic_request(body)["max_tokens"] == 50


def test_request_converts_images() -> None:
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]
    request = to_anthropic_request(
        {"model": "m", "messages": [{"role": "user", "content": content}]}
    )
    blocks = request["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "What is this?"}
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert blocks[2]["source"] == {"type": "url", "url": "https://example.com/cat.png"}


def test_request_rejects_tools_and_unknown_roles() -> None:
    with pytest.raises(GatewayError):
        to_anthropic_request({"model": "m", "messages": [], "tools": [{"type": "function"}]})
    with pytest.raises(GatewayError):
        to_anthropic_request({"model": "m", "messages": [{"role": "tool", "content": "x"}]})


def test_response_is_translated_to_openai_shape() -> None:
    result = from_anthropic_response(
        {
            "id": "msg_1",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Hello"},
            ],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
    )
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hello"}
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_chat_end_to_end_sends_anthropic_headers(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    body = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}

    response = client.post("/v1/chat/completions", json=body, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hi"
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "sk-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(sent.content)["model"] == "claude-opus-5-5"


def test_anthropic_error_is_translated(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )
    )
    body = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}

    response = client.post("/v1/chat/completions", json=body, headers=AUTH)

    assert response.status_code == 429
    assert response.json()["error"] == {
        "message": "slow down",
        "type": "rate_limit_error",
        "param": None,
        "code": None,
    }


ANTHROPIC_STREAM = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5",'
    b'"usage":{"input_tokens":7}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":2}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


def parse_openai_stream(raw: bytes) -> list:
    events = [
        line[len("data: ") :] for line in raw.decode().split("\n") if line.startswith("data: ")
    ]
    return [e if e == "[DONE]" else json.loads(e) for e in events]


def test_stream_is_translated_to_openai_chunks(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    upstream.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, content=ANTHROPIC_STREAM, headers={"content-type": "text/event-stream"}
        )
    )
    body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response = client.post("/v1/chat/completions", json=body, headers=AUTH)
    events = parse_openai_stream(response.content)

    text = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e != "[DONE]" and e["choices"]
    )
    assert text == "Hello"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-3]["choices"][0]["finish_reason"] == "stop"
    assert events[-2]["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert events[-1] == "[DONE]"


def test_stream_error_event_becomes_error_chunk(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    stream = (
        b"event: error\n"
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
    )
    upstream.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=stream))
    body = {"model": "claude", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    events = parse_openai_stream(
        client.post("/v1/chat/completions", json=body, headers=AUTH).content
    )

    assert events[0]["error"]["message"] == "busy"
    assert events[-1] == "[DONE]"


def test_missing_provider_key_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_gateway.app import create_app
    from tests.conftest import make_config

    monkeypatch.delenv("TEST_ANTHROPIC_KEY")
    with pytest.raises(RuntimeError, match="TEST_ANTHROPIC_KEY"):
        with TestClient(create_app(config=make_config(), api_keys=["k"])):
            pass
