"""Locust scenarios. Pick one with the SCENARIO environment variable (see benchmarks/run.py).

direct          call the mock upstream directly (baseline: client + network + mock)
proxy           through the gateway, caching skipped (auth, rate limit, routing, usage)
miss            through the gateway, every prompt unique (exact + semantic lookup, embed, store)
exact_hit       the same prompt every time (exact cache hit)
semantic_hit    rewordings of one cached question (semantic cache hit)
"""

import itertools
import os
import random

from locust import FastHttpUser, constant, task

SCENARIO = os.environ.get("SCENARIO", "proxy")
API_KEY = os.environ.get("GATEWAY_API_KEY", "bench-key")
AUTH = {"Authorization": f"Bearer {API_KEY}"}
REWORDINGS = itertools.cycle(["What's France's capital city?", "what is the capital of France"])
# Each scenario must produce this cache result, or the request counts as a failure,
# so a scenario can't silently measure something else
EXPECTED_CACHE = {"miss": "miss", "exact_hit": "hit", "semantic_hit": "semantic-hit"}


def body(content: str) -> dict:
    return {
        "model": "bench" if SCENARIO != "direct" else "mock-model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }


class GatewayUser(FastHttpUser):
    wait_time = constant(0)  # send the next request as soon as the last one finishes

    @task
    def chat(self) -> None:
        headers = AUTH
        prompt = "What is the capital of France?"
        if SCENARIO == "direct":
            headers = {}
        elif SCENARIO == "proxy":
            headers = {**AUTH, "x-gateway-cache": "skip"}
        elif SCENARIO == "miss":
            # A different number each time: a normal-length question that can never hit
            # (the semantic cache's number check blocks "country 12" matching "country 13")
            prompt = f"What is the capital of country number {random.randrange(10**9)}?"
        elif SCENARIO == "semantic_hit":
            prompt = next(REWORDINGS)

        with self.client.post(
            "/v1/chat/completions", json=body(prompt), headers=headers, catch_response=True
        ) as response:
            expected = EXPECTED_CACHE.get(SCENARIO)
            actual = response.headers.get("x-gateway-cache")
            if expected and actual != expected:
                response.failure(f"expected cache={expected}, got {actual}")
