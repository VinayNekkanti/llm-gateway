import json

import httpx
import respx
from fastapi.testclient import TestClient

from tests.conftest import AUTH, UPSTREAM_URL, chat_response

BODY = {"model": "llama3.2:1b", "messages": [{"role": "user", "content": "hi"}]}


def test_forwards_request_and_returns_upstream_answer(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, json=chat_response("Hi there"))
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hi there"
    # The body we sent upstream is exactly what the client sent us
    assert json.loads(route.calls.last.request.content) == BODY


def test_streaming_passes_chunks_through(client: TestClient, upstream: respx.MockRouter) -> None:
    sse = (
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    upstream.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )

    response = client.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == sse
