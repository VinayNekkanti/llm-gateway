import hashlib
import math
from collections.abc import Iterator

import httpx
import pytest
import redis
import respx
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.cache.semantic import context_hash, likely_different_question, split_request
from tests.conftest import API_KEY, AUTH, OLLAMA_URL, chat_response, make_config


class FakeEmbedder:
    """Deterministic stand-in for the real model: known texts get chosen vectors."""

    def __init__(self) -> None:
        a, b = math.cos(0.2), math.sin(0.2)  # cosine(base, near) = cos(0.2) ~ 0.98
        self.known = {
            "What is the capital of France?": [1.0, 0.0, 0.0],
            "What's France's capital city?": [a, b, 0.0],
            "Tell me a joke.": [0.0, 0.0, 1.0],
            # Very close vector but a different question: the text checks must block this
            "What is the capital of Germany?": [a, b, 0.0],
            # Identical vector to the rewording, so it is the nearest match, but it's a
            # one-word swap ("France's" -> "Spain's") that the text checks reject
            "What's Spain's capital city?": [a, b, 0.0],
        }
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        if text in self.known:
            return self.known[text]
        digest = hashlib.sha256(text.encode()).digest()
        vector = [b - 128 for b in digest[:3]]
        norm = math.sqrt(sum(x * x for x in vector)) or 1
        return [x / norm for x in vector]


def ask(client: TestClient, question: str, **extra: object) -> httpx.Response:
    body = {
        "model": "smart",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        **extra,
    }
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def semantic_client(
    embedder: FakeEmbedder, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> Iterator[TestClient]:
    config = make_config(
        cache={
            "exact": {"enabled": False},
            "semantic": {"enabled": True, "similarity_threshold": 0.9},
        }
    )
    with TestClient(create_app(config=config, api_keys=[API_KEY], embedder=embedder)) as client:
        yield client


def test_reworded_question_is_a_semantic_hit(
    semantic_client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(OLLAMA_URL).mock(
        return_value=httpx.Response(200, json=chat_response("Paris"))
    )

    first = ask(semantic_client, "What is the capital of France?")
    second = ask(semantic_client, "What's France's capital city?")

    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "semantic-hit"
    assert float(second.headers["x-gateway-similarity"]) > 0.9
    assert second.json()["choices"][0]["message"]["content"] == "Paris"
    assert route.call_count == 1


def test_unrelated_question_misses(semantic_client: TestClient, upstream: respx.MockRouter) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "What is the capital of France?")
    assert ask(semantic_client, "Tell me a joke.").headers["x-gateway-cache"] == "miss"
    assert route.call_count == 2


def test_text_checks_block_similar_but_different_question(
    semantic_client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "What is the capital of France?")
    response = ask(semantic_client, "What is the capital of Germany?")
    assert response.headers["x-gateway-cache"] == "miss"
    assert route.call_count == 2


def test_different_system_prompt_never_matches(
    semantic_client: TestClient, upstream: respx.MockRouter
) -> None:
    route = upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "What is the capital of France?")
    body = {
        "model": "smart",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "What's France's capital city?"},
        ],
    }
    response = semantic_client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert response.headers["x-gateway-cache"] == "miss"
    assert route.call_count == 2


def test_expired_answer_is_cleaned_up(
    semantic_client: TestClient, upstream: respx.MockRouter, clean_redis: redis.Redis
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "What is the capital of France?")
    # Simulate the TTL running out on the stored answer (the vector stays behind)
    for key in clean_redis.scan_iter("llmgw:cache:semantic:resp:*"):
        clean_redis.delete(key)

    response = ask(semantic_client, "What's France's capital city?")

    assert response.headers["x-gateway-cache"] == "miss"


def test_high_temperature_skips_embedding(
    semantic_client: TestClient, embedder: FakeEmbedder, upstream: respx.MockRouter
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "What is the capital of France?", temperature=1.0)
    assert embedder.calls == 0


def test_likely_different_question_rules() -> None:
    assert likely_different_question("What is 15% of 200?", "What is 20% of 200?")
    assert likely_different_question("Should I use tabs?", "Should I not use tabs?")
    assert likely_different_question("Convert 10 miles to km", "Convert 10 km to miles")
    assert likely_different_question("Capital of France?", "Capital of Germany?")
    # Real paraphrases pass
    assert not likely_different_question("When did WW2 end?", "What year did World War II end?")
    assert not likely_different_question("How do I exit vim?", "How can I quit the vim editor?")


def test_split_request_needs_a_final_user_text() -> None:
    assert split_request({"messages": [{"role": "assistant", "content": "hi"}]}) is None
    image = [{"type": "image_url", "image_url": {"url": "x"}}]
    assert split_request({"messages": [{"role": "user", "content": image}]}) is None
    question, earlier, params = split_request(  # type: ignore[misc]
        {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "q"}]}
    )
    assert (question, earlier, params) == ("q", [], {"temperature": 0})


def test_context_hash_depends_on_history_and_scope() -> None:
    base = context_hash([], {}, "p/m", "k")
    assert base != context_hash([{"role": "system", "content": "x"}], {}, "p/m", "k")
    assert base != context_hash([], {}, "p/m", "other-key")
    assert base != context_hash([], {"max_tokens": 5}, "p/m", "k")


def test_matches_reads_both_reply_formats() -> None:
    from llm_gateway.cache.semantic import _matches

    assert _matches({b"a": [0.9, b"{}"], b"b": [0.8, None]}) == [
        (b"a", 0.9, b"{}"),
        (b"b", 0.8, None),
    ]
    assert _matches([b"a", b"0.9", b"{}", b"b", b"0.8", None]) == [
        (b"a", 0.9, b"{}"),
        (b"b", 0.8, None),
    ]
    assert _matches({}) == []
    assert _matches(None) == []


def test_rejected_nearest_match_does_not_hide_a_valid_one(
    semantic_client: TestClient, upstream: respx.MockRouter
) -> None:
    # Found by the benchmark: only checking the single nearest neighbour meant a rejected
    # look-alike could hide a valid match right behind it
    upstream.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(200, json=chat_response("Paris")),
            httpx.Response(200, json=chat_response("Madrid")),
        ]
    )
    ask(semantic_client, "What is the capital of France?")
    ask(semantic_client, "What's Spain's capital city?")

    response = ask(semantic_client, "What's France's capital city?")

    assert response.headers["x-gateway-cache"] == "semantic-hit"
    assert response.json()["choices"][0]["message"]["content"] == "Paris"


def test_long_questions_skip_semantic_cache(
    semantic_client: TestClient, embedder: FakeEmbedder, upstream: respx.MockRouter
) -> None:
    upstream.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=chat_response()))
    ask(semantic_client, "word " * 100)  # 500 characters, over the 300 default
    assert embedder.calls == 0
