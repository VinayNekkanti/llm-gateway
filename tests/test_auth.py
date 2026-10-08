import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.auth import key_id, load_api_keys
from tests.conftest import AUTH, UPSTREAM_URL, chat_response

BODY = {"model": "llama3.2:1b", "messages": [{"role": "user", "content": "hi"}]}


def test_health_needs_no_key(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_key_is_rejected(client: TestClient, upstream: respx.MockRouter) -> None:
    route = upstream.post(UPSTREAM_URL).mock(return_value=httpx.Response(200, json=chat_response()))

    response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"
    # Rejected requests must never reach (or cost money at) the LLM
    assert not route.called


def test_wrong_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_right_key_is_accepted(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(UPSTREAM_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    response = client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert response.status_code == 200


def test_load_api_keys_splits_and_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", " a, b ,,c ")
    assert load_api_keys() == ["a", "b", "c"]


def test_load_api_keys_fails_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GATEWAY_API_KEYS"):
        load_api_keys()


def test_key_id_does_not_leak_the_key() -> None:
    assert key_id("secret-value").startswith("key_")
    assert "secret" not in key_id("secret-value")
