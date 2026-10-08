import httpx
import respx
import structlog
from fastapi.testclient import TestClient

from tests.conftest import AUTH, FLAKY_URL, OLLAMA_URL, chat_response


def ask(client: TestClient, model: str = "smart", **headers: str) -> httpx.Response:
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    return client.post("/v1/chat/completions", json=body, headers={**AUTH, **headers})


def metric(client: TestClient, line_start: str) -> float:
    """Read one value from the Prometheus text output."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_every_response_has_a_request_id(client: TestClient) -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert first and second and first != second


def test_incoming_request_id_is_kept(client: TestClient) -> None:
    # Lets a caller trace one request across their logs and ours
    assert client.get("/health", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"


def test_request_log_line(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    with structlog.testing.capture_logs() as logs:
        ask(client)
    line = next(entry for entry in logs if entry["event"] == "request")
    assert line["route"] == "/v1/chat/completions"
    assert line["status"] == 200
    assert line["model"] == "smart"
    assert line["key_id"].startswith("key_")
    assert "test-key" not in str(line)  # the raw API key never appears in logs


def test_metrics_count_requests_and_tokens(client: TestClient, upstream: respx.MockRouter) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    route = 'llmgw_http_requests_total{method="POST",route="/v1/chat/completions",status="200"}'
    tokens = 'llmgw_tokens_total{kind="prompt",model="smart"}'
    before_requests, before_tokens = metric(client, route), metric(client, tokens)

    ask(client)

    assert metric(client, route) == before_requests + 1
    assert metric(client, tokens) == before_tokens + 5  # chat_response() uses 5 prompt tokens


def test_unknown_paths_share_one_metric_label(client: TestClient) -> None:
    client.get("/random-1")
    client.get("/random-2")
    text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "random-1" not in text  # raw paths would create unbounded time series


def test_fallback_and_upstream_failures_are_counted(
    client: TestClient, upstream: respx.MockRouter
) -> None:
    upstream.post(FLAKY_URL).mock(return_value=httpx.Response(503, text="down"))
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    fallback = 'llmgw_fallbacks_total{from_model="primary",to_model="smart"}'
    failed = 'llmgw_upstream_attempts_total{model="primary",outcome="503",provider="flaky"}'
    before_fallback, before_failed = metric(client, fallback), metric(client, failed)

    with structlog.testing.capture_logs() as logs:
        ask(client, "primary")

    assert metric(client, fallback) == before_fallback + 1
    assert metric(client, failed) == before_failed + 3  # three attempts before falling back
    assert any(entry["event"] == "falling_back" for entry in logs)


def test_metrics_endpoint_is_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "llmgw_http_request_duration_seconds_bucket" in response.text
