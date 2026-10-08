import httpx
import respx
from fastapi.testclient import TestClient

from tests.conftest import AUTH, FLAKY_URL, OLLAMA_URL, chat_response


def ask(client: TestClient, model: str, **extra: object) -> httpx.Response:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def test_temporary_error_is_retried(client: TestClient, upstream: respx.MockRouter) -> None:
    route = upstream.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.ConnectError("refused"),
            httpx.Response(200, json=chat_response("finally")),
        ]
    )

    response = ask(client, "smart")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "finally"
    assert route.call_count == 3


def test_client_error_is_not_retried(client: TestClient, upstream: respx.MockRouter) -> None:
    error = {"error": {"message": "bad", "type": "invalid_request_error"}}
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(400, json=error))

    assert ask(client, "smart").status_code == 400
    assert route.call_count == 1


def test_gives_up_after_max_attempts(client: TestClient, upstream: respx.MockRouter) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(500, text="down"))

    assert ask(client, "smart").status_code == 500
    assert route.call_count == 3  # max_attempts in the test config


def test_falls_back_when_primary_is_down(client: TestClient, upstream: respx.MockRouter) -> None:
    primary = upstream.post(FLAKY_URL).mock(side_effect=httpx.ConnectError("refused"))
    backup = upstream.post(OLLAMA_URL).mock(
        return_value=httpx.Response(200, json=chat_response("from backup"))
    )

    response = ask(client, "primary")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from backup"
    assert response.headers["x-gateway-model"] == "smart"
    assert primary.call_count == 3  # retried first, then fell back
    assert backup.call_count == 1


def test_no_fallback_for_client_errors(client: TestClient, upstream: respx.MockRouter) -> None:
    error = {"error": {"message": "bad", "type": "invalid_request_error"}}
    upstream.post(FLAKY_URL).mock(return_value=httpx.Response(400, json=error))
    backup = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))

    assert ask(client, "primary").status_code == 400
    assert not backup.called


def test_all_models_down_returns_last_error(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(FLAKY_URL).mock(side_effect=httpx.ConnectError("refused"))
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(503, text="also down"))

    response = ask(client, "primary")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "also down"


def test_streaming_falls_back_before_first_byte(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    upstream.post(FLAKY_URL).mock(return_value=httpx.Response(503, text="busy"))
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, content=sse))

    response = ask(client, "primary", stream=True)

    assert response.status_code == 200
    assert response.headers["x-gateway-model"] == "smart"
    assert response.content == sse


def test_served_by_header_on_normal_requests(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    assert ask(client, "smart").headers["x-gateway-model"] == "smart"
