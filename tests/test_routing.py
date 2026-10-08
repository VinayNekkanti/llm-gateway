import json

import httpx
import respx
from fastapi.testclient import TestClient

from tests.conftest import AUTH, OLLAMA_URL, chat_response


def ask(client: TestClient, model: str, **extra: object) -> httpx.Response:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def test_alias_is_replaced_with_provider_model(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))

    assert ask(client, "smart").status_code == 200
    assert json.loads(route.calls.last.request.content)["model"] == "llama3.2"


def test_unknown_model_is_404(client: TestClient) -> None:
    response = ask(client, "gpt-9")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_drop_params_removes_fields(client: TestClient, upstream: respx.MockRouter) -> None:
    route = upstream.post("http://anthropic.test/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    ask(client, "claude", temperature=0.2)
    assert "temperature" not in json.loads(route.calls.last.request.content)


def test_upstream_error_is_passed_through(client: TestClient, upstream: respx.MockRouter) -> None:
    error = {"error": {"message": "bad input", "type": "invalid_request_error"}}
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(400, json=error))

    response = ask(client, "llama3.2:1b")

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad input"


def test_non_json_upstream_error_is_wrapped(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(500, text="boom"))

    response = ask(client, "llama3.2:1b")

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "boom"


def test_unreachable_provider_is_502(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError("refused"))

    response = ask(client, "llama3.2:1b")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_provider_timeout_is_504(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    assert ask(client, "llama3.2:1b").status_code == 504


def test_streaming_error_before_first_byte_is_normal_error(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(503, text="overloaded"))

    response = ask(client, "llama3.2:1b", stream=True)

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "overloaded"


def test_streaming_unreachable_provider_is_502(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    upstream.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError("refused"))
    assert ask(client, "llama3.2:1b", stream=True).status_code == 502


def test_list_models(client: TestClient) -> None:
    response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert ids == ["claude", "llama3.2:1b", "primary", "smart"]


def test_list_models_needs_key(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401


def test_bad_requests_are_400(client: TestClient) -> None:
    send = client.post
    assert send("/v1/chat/completions", content=b"not json", headers=AUTH).status_code == 400
    assert send("/v1/chat/completions", json={"messages": []}, headers=AUTH).status_code == 400
    no_messages = {"model": "smart", "messages": []}
    assert send("/v1/chat/completions", json=no_messages, headers=AUTH).status_code == 400
