"""Cache hit rate, cost saved and wrong answers on a realistic mix of traffic.

Traffic is built from the labeled pairs in eval/ (both the tuning and held-out sets):
  - new questions                  (first time asked: must miss)
  - exact repeats                  (should be exact hits)
  - rewordings of asked questions  ("same" pairs: may be semantic hits)
  - look-alikes of asked questions ("different" pairs: must never get a cached answer)
The mock upstream answers "Answer to: <question>", so every cached answer can be checked.
"""

import json
import random
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).parent.parent


def load_pairs() -> list[dict[str, Any]]:
    pairs = []
    for name in ("semantic_cache_pairs.jsonl", "semantic_cache_pairs_holdout.jsonl"):
        lines = (ROOT / "eval" / name).read_text().splitlines()
        pairs += [json.loads(line) for line in lines if line.strip()]
    return pairs


def build_traffic(pairs: list[dict[str, Any]], total: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    same = [p for p in pairs if p["same"]]
    different = [p for p in pairs if not p["same"]]
    asked: list[str] = []
    traffic: list[str] = []
    unused = [p["a"] for p in pairs]
    rng.shuffle(unused)
    while len(traffic) < total:
        roll = rng.random()
        asked_set = set(asked)
        rewordings = [p["b"] for p in same if p["a"] in asked_set]
        lookalikes = [p["b"] for p in different if p["a"] in asked_set]
        if roll < 0.35 and asked:
            question = rng.choice(asked)  # exact repeat
        elif roll < 0.55 and rewordings:
            question = rng.choice(rewordings)
        elif roll < 0.65 and lookalikes:
            question = rng.choice(lookalikes)
        elif unused:
            question = unused.pop()
        else:
            question = f"What is the capital of country number {rng.randrange(10**9)}?"
        traffic.append(question)
        asked.append(question)
    return traffic


def equivalent(pairs: list[dict[str, Any]]) -> set[frozenset[str]]:
    return {frozenset((p["a"], p["b"])) for p in pairs if p["same"]}


def run(base_url: str, api_key: str, total: int = 500) -> dict[str, Any]:
    pairs = load_pairs()
    same_meaning = equivalent(pairs)
    traffic = build_traffic(pairs, total)
    counts = {"hit": 0, "semantic-hit": 0, "miss": 0, "skip": 0}
    wrong: list[tuple[str, str]] = []
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=30) as client:
        for question in traffic:
            body = {
                "model": "bench",
                "temperature": 0,
                "messages": [{"role": "user", "content": question}],
            }
            response = client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            counts[response.headers["x-gateway-cache"]] += 1
            answered = response.json()["choices"][0]["message"]["content"].removeprefix(
                "Answer to: "
            )
            if answered != question and frozenset((answered, question)) not in same_meaning:
                wrong.append((question, answered))
        usage = client.get("/usage", params={"days": 1}).json()["totals"]

    paid, saved = usage["cost_usd"], usage["cost_saved_usd"]
    return {
        "requests": total,
        "exact_hits": counts["hit"],
        "semantic_hits": counts["semantic-hit"],
        "misses": counts["miss"],
        "wrong_answers": len(wrong),
        "wrong_examples": wrong[:5],
        "cost_without_cache": paid + saved,
        "cost_with_cache": paid,
        "saved_fraction": saved / (paid + saved) if paid + saved else 0.0,
    }
