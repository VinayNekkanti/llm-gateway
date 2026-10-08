"""Semantic cache: reuse an answer when a new question *means* the same as an earlier one.

How it works:
1. Split the request into the latest user message (the "question") and everything else
   (system prompt, earlier turns, parameters, model) - the "context".
2. The context must match exactly, so it's hashed. "Summarize this" with a different
   document attached is a different request, however similar the words.
3. The question is turned into an embedding (384 numbers capturing its meaning) and stored in
   a Redis 8 vector set, tagged with the context hash.
4. A new request searches only entries with the same context hash, and reuses the closest
   answer if its cosine similarity is above the configured threshold.
"""

import asyncio
import hashlib
import json
import re
import struct
import uuid
from typing import Any, Protocol

from redis.asyncio import Redis

from llm_gateway.cache import NOT_PART_OF_KEY
from llm_gateway.redis_store import RedisStore

INDEX_KEY = "llmgw:cache:semantic:index"
RESPONSE_PREFIX = "llmgw:cache:semantic:resp:"


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Runs a small embedding model (BAAI/bge-small-en-v1.5) locally on the CPU."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # imported here: loading the library is slow

        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed(self, text: str) -> list[float]:
        vector = next(iter(self.model.embed([text])))
        return [float(x) for x in vector]


def split_request(body: dict[str, Any]) -> tuple[str, list[Any], dict[str, Any]] | None:
    """(question, earlier messages, other params), or None if the last message isn't a user text."""
    messages = body.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    if not isinstance(last.get("content"), str):
        return None  # images or other parts: don't guess what "similar" means
    params = {k: v for k, v in body.items() if k not in NOT_PART_OF_KEY | {"model", "messages"}}
    return last["content"], messages[:-1], params


def context_hash(earlier: list[Any], params: dict[str, Any], model_id: str, scope: str) -> str:
    canonical = json.dumps(
        {"model": model_id, "scope": scope, "messages": earlier, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


NEGATIONS = {"not", "no", "never", "without", "dont", "doesnt", "isnt", "cant", "wont", "shouldnt"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def likely_different_question(a: str, b: str) -> bool:
    """Cheap text checks for edits that change the answer but barely move the embedding.

    Measured on eval/semantic_cache_pairs.jsonl: embeddings score "10 miles to km" vs
    "10 km to miles" at 0.99, higher than most real paraphrases. These rules catch that.
    """
    ta, tb = _tokens(a), _tokens(b)
    # Different numbers: "15% of 200" vs "20% of 200"
    numbers_a = {t for t in ta if t.isdigit()}
    numbers_b = {t for t in tb if t.isdigit()}
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return True
    # One side negated: "should I use tabs" vs "should I not use tabs"
    if bool(NEGATIONS & set(ta)) != bool(NEGATIONS & set(tb)):
        return True
    # Same words, different order: "string to integer" vs "integer to string"
    if sorted(ta) == sorted(tb) and ta != tb:
        return True
    # Exactly one word swapped: "capital of France" vs "capital of Germany"
    if len(ta) == len(tb) and sum(x != y for x, y in zip(ta, tb, strict=True)) == 1:
        return True
    return False


def to_fp32(vector: list[float]) -> bytes:
    # Redis accepts vectors as packed 32-bit floats: smaller and faster than text numbers
    return struct.pack(f"<{len(vector)}f", *vector)


# How many nearest neighbours to check. The nearest one can be a look-alike that the text
# checks reject; a valid match may be just behind it.
CANDIDATES = 5


def _matches(found: Any) -> list[tuple[bytes, float, bytes | None]]:
    """VSIM's reply, nearest first: redis-py gives {element: [score, attributes]};
    raw Redis gives a flat list [element, score, attributes, ...]."""
    if not found:
        return []
    if isinstance(found, dict):
        return [(element, float(score), attrs) for element, (score, attrs) in found.items()]
    return [(found[i], float(found[i + 1]), found[i + 2]) for i in range(0, len(found) - 2, 3)]


class SemanticCache:
    def __init__(
        self,
        store: RedisStore,
        embedder: Embedder,
        threshold: float,
        ttl_seconds: int,
        max_question_chars: int = 300,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_question_chars = max_question_chars

    async def embed(self, question: str) -> list[float]:
        # The model runs on the CPU; a worker thread keeps the event loop free for other requests
        return await asyncio.to_thread(self.embedder.embed, question)

    async def lookup(
        self, question: str, vector: list[float], ctx: str
    ) -> tuple[dict[str, Any], float] | None:
        """Closest cached answer with the same context, if similar enough: (answer, similarity)."""

        async def search(redis: Redis) -> Any:
            return await redis.execute_command(
                "VSIM", INDEX_KEY, "FP32", to_fp32(vector),
                "WITHSCORES", "WITHATTRIBS", "COUNT", CANDIDATES, "FILTER", f'.ctx == "{ctx}"',
            )  # fmt: skip

        for element, score, raw_attributes in _matches(await self.store.call(search)):
            # Redis scores are (1 + cosine) / 2, from 0 (opposite) to 1 (identical)
            similarity = 2 * score - 1
            if similarity < self.threshold:
                return None  # results are sorted, so the rest are even less similar
            attributes = json.loads(raw_attributes) if raw_attributes else {}
            if likely_different_question(question, attributes.get("q", "")):
                continue  # a look-alike with a different answer: try the next candidate

            raw = await self._answer(element)
            if raw is None:
                # The answer expired (TTL) but its vector is still indexed: clean it up lazily
                await self._forget(element)
                continue
            answer: dict[str, Any] = json.loads(raw)
            return answer, similarity
        return None

    async def _answer(self, element: bytes) -> bytes | str | None:
        return await self.store.call(lambda r: r.get(RESPONSE_PREFIX + element.decode()))

    async def _forget(self, element: bytes) -> None:
        await self.store.call(lambda r: r.execute_command("VREM", INDEX_KEY, element))

    async def store_answer(
        self, question: str, vector: list[float], ctx: str, result: dict[str, Any]
    ) -> None:
        element = uuid.uuid4().hex

        async def save(redis: Redis) -> None:
            # Answer first (with TTL), then the vector that points at it
            await redis.set(RESPONSE_PREFIX + element, json.dumps(result), ex=self.ttl_seconds)
            await redis.execute_command(
                "VADD", INDEX_KEY, "FP32", to_fp32(vector), element,
                "SETATTR", json.dumps({"ctx": ctx, "q": question}),
            )  # fmt: skip

        await self.store.call(save)
