import httpx
import redis
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.cache import is_cacheable_request, is_cacheable_response, request_fingerprint
from tests.conftest import API_KEY, AUTH, OLLAMA_URL, chat_response, make_config

BODY = {"model": "smart", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}


def ask(client: TestClient, body: dict = BODY, headers: dict = AUTH) -> httpx.Response:
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_repeat_request_is_served_from_cache(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))

    first = ask(client)
    second = ask(client)

    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.json() == first.json()
    assert route.call_count == 1  # the second answer never touched the model


def test_skip_header_bypasses_cache(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))

    ask(client)
    response = ask(client, headers={**AUTH, "x-gateway-cache": "skip"})

    assert response.headers["x-gateway-cache"] == "skip"
    assert route.call_count == 2


def test_high_temperature_is_not_cached(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    body = {**BODY, "temperature": 0.9}

    ask(client, body)
    response = ask(client, body)

    assert response.headers["x-gateway-cache"] == "skip"
    assert route.call_count == 2


def test_errors_are_never_cached(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    route = upstream.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "bad", "type": "x"}}),
            httpx.Response(200, json=chat_response()),
        ]
    )

    assert ask(client).status_code == 400
    assert ask(client).status_code == 200
    assert route.call_count == 2


def test_aliases_share_cache_entries(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    # "llama3.2:1b" is also reachable by that name; both names point at the same real model
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(client, {**BODY, "model": "llama3.2:1b"})
    ask(client, {**BODY, "model": "llama3.2:1b"})
    assert route.call_count == 1


def test_keys_do_not_share_cache_by_default(
    config: object, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    app = create_app(config=make_config(), api_keys=[API_KEY, "other-key"])
    with TestClient(app) as client:
        ask(client)
        response = ask(client, headers={"Authorization": "Bearer other-key"})
    assert response.headers["x-gateway-cache"] == "miss"
    assert route.call_count == 2


def test_gateway_keeps_working_when_redis_is_down(upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    config = make_config(redis={"url": "redis://localhost:1/0", "timeout_seconds": 0.1})
    with TestClient(create_app(config=config, api_keys=[API_KEY])) as client:
        assert ask(client).status_code == 200
        assert ask(client).status_code == 200


def test_fingerprint_ignores_key_order_and_delivery_fields() -> None:
    a = {"model": "x", "messages": [], "temperature": 0, "stream": False}
    b = {"temperature": 0, "messages": [], "model": "y", "user": "bob"}
    assert request_fingerprint(a, "p/m", "k") == request_fingerprint(b, "p/m", "k")
    assert request_fingerprint(a, "p/m", "k") != request_fingerprint(a, "p/m", "other")
    assert request_fingerprint(a, "p/m", "k") != request_fingerprint(a, "p/other", "k")


def test_cacheable_rules() -> None:
    assert is_cacheable_request({"temperature": 0}, 0.3)
    assert not is_cacheable_request({}, 0.3)  # default temperature is 1.0
    assert not is_cacheable_request({"temperature": 0, "n": 2}, 0.3)
    assert not is_cacheable_request({"temperature": 0, "tools": [{}]}, 0.3)
    assert is_cacheable_response(chat_response())
    cut_off = chat_response()
    cut_off["choices"][0]["finish_reason"] = "length"
    assert not is_cacheable_response(cut_off)
    assert not is_cacheable_response({"choices": []})
