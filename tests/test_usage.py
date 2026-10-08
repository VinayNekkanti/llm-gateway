import json
from collections.abc import AsyncIterator

import httpx
import redis
import respx
from fastapi.testclient import TestClient

from llm_gateway.config import ModelConfig, Pricing
from llm_gateway.usage import cost_micro_usd, track_stream
from tests.conftest import AUTH, OLLAMA_URL, chat_response


def usage_response(prompt: int, completion: int) -> dict:
    result = chat_response()
    result["usage"] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    return result


def ask(client: TestClient, model: str = "smart", **extra: object) -> httpx.Response:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def test_cost_math() -> None:
    model = ModelConfig(
        provider="p", model="m", pricing=Pricing(input_per_million=2, output_per_million=10)
    )
    # 1,000 input tokens at $2/M = $0.002; 500 output at $10/M = $0.005; total $0.007
    assert cost_micro_usd(model, 1000, 500) == 7000
    assert cost_micro_usd(ModelConfig(provider="p", model="m"), 1000, 500) == 0


def test_usage_is_recorded_and_reported(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=usage_response(1000, 500)))

    response = ask(client)
    ask(client)
    report = client.get("/usage", headers=AUTH).json()

    assert response.headers["x-gateway-cost-usd"] == "0.007000"
    assert report["totals"]["requests"] == 2
    assert report["totals"]["prompt_tokens"] == 2000
    assert report["totals"]["completion_tokens"] == 1000
    assert report["totals"]["cost_usd"] == 0.014
    assert report["by_model"]["smart"]["requests"] == 2
    assert len(report["by_day"]) == 1


def test_cache_hits_count_as_savings(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=usage_response(1000, 500)))

    ask(client, temperature=0)
    ask(client, temperature=0)  # exact cache hit
    totals = client.get("/usage", headers=AUTH).json()["totals"]

    assert totals["requests"] == 2
    assert totals["cache_hits"] == 1
    assert totals["cost_usd"] == 0.007  # only the first request was paid for
    assert totals["cost_saved_usd"] == 0.007


def test_streaming_usage_is_recorded_and_hidden_unless_asked(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    sse = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50}}\n\n'
        b"data: [DONE]\n\n"
    )
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, content=sse))

    response = ask(client, stream=True)
    sent = json.loads(route.calls.last.request.content)
    totals = client.get("/usage", headers=AUTH).json()["totals"]

    assert sent["stream_options"] == {"include_usage": True}  # gateway asked for usage
    assert b"usage" not in response.content  # but the client didn't, so it's removed
    assert b"[DONE]" in response.content
    assert totals["prompt_tokens"] == 100
    assert totals["completion_tokens"] == 50


def test_streaming_usage_kept_when_client_asks(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    sse = (
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        b"data: [DONE]\n\n"
    )
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, content=sse))

    response = ask(client, stream=True, stream_options={"include_usage": True})

    assert b'"usage"' in response.content


def test_usage_is_per_key(
    client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    from llm_gateway.app import create_app
    from tests.conftest import API_KEY, make_config

    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=usage_response(10, 10)))
    with TestClient(create_app(config=make_config(), api_keys=[API_KEY, "other"])) as c:
        ask(c)
        other = c.get("/usage", headers={"Authorization": "Bearer other"}).json()
    assert other["totals"]["requests"] == 0


def test_usage_needs_key_and_valid_days(client: TestClient, clean_redis: redis.Redis) -> None:
    assert client.get("/usage").status_code == 401
    assert client.get("/usage?days=0", headers=AUTH).status_code == 422
    assert client.get("/usage?days=7", headers=AUTH).json()["totals"]["requests"] == 0


def test_usage_unavailable_when_redis_down(upstream: respx.MockRouter) -> None:
    from llm_gateway.app import create_app
    from tests.conftest import API_KEY, make_config

    config = make_config(redis={"url": "redis://localhost:1/0", "timeout_seconds": 0.1})
    with TestClient(create_app(config=config, api_keys=[API_KEY])) as c:
        assert c.get("/usage", headers=AUTH).status_code == 503


async def test_track_stream_handles_non_json_and_missing_usage() -> None:
    seen: list = []

    async def chunks() -> AsyncIterator[bytes]:
        yield b"data: not-json\n\ndata: [DONE]\n\n"

    async def on_usage(usage: dict | None) -> None:
        seen.append(usage)

    out = [c async for c in track_stream(chunks(), False, on_usage)]
    assert out == [b"data: not-json\n\n", b"data: [DONE]\n\n"]
    assert seen == [None]
