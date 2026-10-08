"""Token and cost tracking per API key, per model, per day (UTC), stored in Redis hashes.

One hash per key per day: llmgw:usage:<key_id>:<YYYY-MM-DD>
with fields like "fast|prompt_tokens". Costs are stored as integer micro-dollars
(millionths of a dollar) because adding up floats slowly drifts.
"""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from redis.asyncio import Redis

from llm_gateway.config import ModelConfig
from llm_gateway.redis_store import RedisStore
from llm_gateway.sse import iter_sse_events, sse_data

PREFIX = "llmgw:usage:"
FIELDS = ("requests", "cache_hits", "prompt_tokens", "completion_tokens", "cost_micro_usd")
SAVED = "saved_micro_usd"
RETENTION_DAYS = 90


def cost_micro_usd(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> int:
    """Estimated cost from config prices (USD per million tokens), in micro-dollars."""
    if model.pricing is None:
        return 0
    # tokens * ($ per 1M tokens) / 1M = $, and $ * 1M = micro-dollars: the millions cancel
    return round(
        prompt_tokens * model.pricing.input_per_million
        + completion_tokens * model.pricing.output_per_million
    )


def token_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    usage = usage or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


class UsageTracker:
    def __init__(self, store: RedisStore, today: Callable[[], date] | None = None) -> None:
        self.store = store
        self.today = today or (lambda: datetime.now(UTC).date())

    async def record(
        self,
        key_id: str,
        model_name: str,
        model: ModelConfig,
        usage: dict[str, Any] | None,
        cache_hit: bool = False,
    ) -> int:
        """Add one request to today's totals. Returns its cost in micro-dollars."""
        prompt_tokens, completion_tokens = token_counts(usage)
        cost = cost_micro_usd(model, prompt_tokens, completion_tokens)
        values = {
            "requests": 1,
            "cache_hits": int(cache_hit),
            # A cache hit costs nothing; what it would have cost is counted as saved
            "prompt_tokens": 0 if cache_hit else prompt_tokens,
            "completion_tokens": 0 if cache_hit else completion_tokens,
            "cost_micro_usd": 0 if cache_hit else cost,
            SAVED: cost if cache_hit else 0,
        }
        key = f"{PREFIX}{key_id}:{self.today().isoformat()}"

        async def write(redis: Redis) -> None:
            pipe = redis.pipeline(transaction=True)
            for name, amount in values.items():
                if amount:
                    pipe.hincrby(key, f"{model_name}|{name}", amount)
            pipe.expire(key, RETENTION_DAYS * 24 * 3600)
            await pipe.execute()

        await self.store.call(write)
        return 0 if cache_hit else cost

    async def report(self, key_id: str, days: int) -> dict[str, Any] | None:
        """Totals, per-model and per-day usage for the last `days` days (None if Redis is down)."""
        end = self.today()
        dates = [end - timedelta(days=offset) for offset in range(days - 1, -1, -1)]

        async def read(redis: Redis) -> list[Any]:
            pipe = redis.pipeline(transaction=False)
            for day in dates:
                pipe.hgetall(f"{PREFIX}{key_id}:{day.isoformat()}")
            result: list[Any] = await pipe.execute()
            return result

        rows = await self.store.call(read)
        if rows is None:
            return None

        totals = _empty()
        by_model: dict[str, dict[str, int]] = {}
        by_day = []
        for day, row in zip(dates, rows, strict=True):
            day_totals = _empty()
            for raw_field, raw_value in row.items():
                model_name, name = raw_field.decode().rsplit("|", 1)
                value = int(raw_value)
                for bucket in (totals, day_totals, by_model.setdefault(model_name, _empty())):
                    bucket[name] = bucket.get(name, 0) + value
            if day_totals["requests"]:
                by_day.append({"date": day.isoformat(), **_dollars(day_totals)})

        return {
            "key_id": key_id,
            "from": dates[0].isoformat(),
            "to": end.isoformat(),
            "totals": _dollars(totals),
            "by_model": {name: _dollars(values) for name, values in sorted(by_model.items())},
            "by_day": by_day,
        }


def _empty() -> dict[str, int]:
    return dict.fromkeys((*FIELDS, SAVED), 0)


def _dollars(values: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {name: values.get(name, 0) for name in FIELDS if name != "cost_micro_usd"}
    out["cost_usd"] = values.get("cost_micro_usd", 0) / 1_000_000
    out["cost_saved_usd"] = values.get(SAVED, 0) / 1_000_000
    return out


async def track_stream(
    chunks: AsyncIterator[bytes],
    client_wants_usage: bool,
    on_usage: Callable[[dict[str, Any] | None], Awaitable[None]],
) -> AsyncIterator[bytes]:
    """Pass a stream through while reading the token usage from its final chunk.

    The gateway always asks the provider for usage (stream_options.include_usage). If the
    client didn't ask for it, the usage-only chunk is removed so the client sees exactly
    what it requested.
    """
    usage: dict[str, Any] | None = None
    try:
        async for _event, data in iter_sse_events(chunks):
            if data != "[DONE]":
                try:
                    parsed = json.loads(data)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("usage"):
                    usage = parsed["usage"]
                    if not parsed.get("choices") and not client_wants_usage:
                        continue  # usage-only chunk the client didn't ask for
            yield sse_data(data)
    finally:
        await on_usage(usage)
