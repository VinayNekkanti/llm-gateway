import hashlib
import json
from typing import Any

# Fields that change how the answer is delivered or tracked, not what the answer is
NOT_PART_OF_KEY = {"stream", "stream_options", "user", "metadata", "store"}


def request_fingerprint(body: dict[str, Any], model_id: str, scope: str) -> str:
    """A hash that is the same for two requests only if they should get the same answer.

    model_id is the real provider/model (so the aliases "fast" and "llama3.2:1b" share entries).
    scope is the API key id, or "shared" when the cache is shared across keys.
    """
    relevant = {k: v for k, v in body.items() if k not in NOT_PART_OF_KEY and k != "model"}
    # sort_keys: {"a":1,"b":2} and {"b":2,"a":1} must hash the same
    canonical = json.dumps(
        {"model": model_id, "scope": scope, "request": relevant},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def is_cacheable_request(body: dict[str, Any], max_temperature: float) -> bool:
    # OpenAI's default temperature is 1.0: high randomness, so a cached answer would be wrong
    # in spirit. Only cache requests that ask for (near-)deterministic answers.
    temperature = body.get("temperature", 1.0)
    if not isinstance(temperature, int | float) or temperature > max_temperature:
        return False
    # n > 1 asks for several different answers; tools make answers depend on outside state
    return body.get("n", 1) == 1 and not body.get("tools")


def is_cacheable_response(result: dict[str, Any]) -> bool:
    # Never cache errors or cut-off answers
    choices = result.get("choices") or []
    return bool(choices) and all(c.get("finish_reason") == "stop" for c in choices)
