"""Run all benchmarks and write benchmarks/results.md.

Needs Redis on localhost:6379. Starts the mock upstream and the gateway itself.
Run: uv run python benchmarks/run.py
"""

import csv
import itertools
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import redis

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.cache_workload import run as run_cache_workload  # noqa: E402

ROOT = Path(__file__).parent.parent
MOCK_PORT, GATEWAY_PORT = 9100, 8100
API_KEY = "bench-key"
LATENCY_REQUESTS = int(os.environ.get("LATENCY_REQUESTS", "2000"))
THROUGHPUT_SECONDS = int(os.environ.get("THROUGHPUT_SECONDS", "20"))
THROUGHPUT_USERS = int(os.environ.get("THROUGHPUT_USERS", "50"))


def start(command: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT
    )


def wait_for(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise RuntimeError(f"{url} did not come up")


def locust(scenario: str, users: int, seconds: int, host: str, out: Path) -> dict[str, float]:
    prefix = out / scenario
    subprocess.run(
        [
            sys.executable, "-m", "locust", "-f", "benchmarks/locustfile.py", "--headless",
            "--users", str(users), "--spawn-rate", str(users), "--run-time", f"{seconds}s",
            "--host", host, "--csv", str(prefix), "--only-summary", "--loglevel", "ERROR",
        ],
        cwd=ROOT,
        env={**os.environ, "SCENARIO": scenario, "GATEWAY_API_KEY": API_KEY},
        # Locust exits with code 1 if any request failed (e.g. connections cut when the
        # timed run stops); failures are reported in the results instead
        check=False,
        capture_output=True,
    )  # fmt: skip
    with open(f"{prefix}_stats.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    return {
        "requests": float(row["Request Count"]),
        "failures": float(row["Failure Count"]),
        "p50": float(row["50%"]),
        "p95": float(row["95%"]),
        "p99": float(row["99%"]),
        "rps": float(row["Requests/s"]),
    }


REWORDINGS = ["What's France's capital city?", "what is the capital of France"]
QUESTION = "What is the capital of France?"


def scenario_request(name: str, counter: "itertools.count[int]") -> tuple[dict, dict]:
    """(body, headers) for one request of a scenario; mirrors benchmarks/locustfile.py."""
    auth = {"Authorization": f"Bearer {API_KEY}"}
    content = QUESTION
    if name == "miss":
        content = f"What is the capital of country number {uuid.uuid4().int % 10**9}?"
    elif name == "semantic_hit":
        content = REWORDINGS[next(counter) % len(REWORDINGS)]
    body = {
        "model": "mock-model" if name == "direct" else "bench",
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    if name == "direct":
        return body, {}
    if name == "proxy":
        return body, {**auth, "x-gateway-cache": "skip"}
    return body, auth


def measure_latency(name: str, host: str) -> dict[str, float]:
    """One request at a time, timed with a high-resolution clock (Locust rounds to 1 ms)."""
    counter = itertools.count()
    timings: list[float] = []
    failures = 0
    with httpx.Client(base_url=host) as client:
        for i in range(LATENCY_REQUESTS + 100):
            body, headers = scenario_request(name, counter)
            start = time.perf_counter()
            response = client.post("/v1/chat/completions", json=body, headers=headers)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i < 100:
                continue  # warm-up: connections, caches, first-call costs
            expected = {"miss": "miss", "exact_hit": "hit", "semantic_hit": "semantic-hit"}
            wrong_path = (
                name in expected and response.headers.get("x-gateway-cache") != expected[name]
            )
            if response.status_code != 200 or wrong_path:
                failures += 1
            timings.append(elapsed_ms)
    cuts = statistics.quantiles(timings, n=100)
    return {"p50": cuts[49], "p95": cuts[94], "p99": cuts[98], "failures": failures}


def warm_up(scenario_requests: list[dict]) -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client(base_url=f"http://127.0.0.1:{GATEWAY_PORT}", headers=headers) as client:
        for body in scenario_requests:
            client.post("/v1/chat/completions", json=body).raise_for_status()


def main() -> None:
    redis.Redis.from_url("redis://localhost:6379/2").flushdb()
    out = Path(tempfile.mkdtemp(prefix="llmgw-bench-"))
    env = {
        **os.environ,
        "GATEWAY_CONFIG": str(ROOT / "benchmarks" / "config.yaml"),
        "GATEWAY_API_KEYS": API_KEY,
    }
    mock = start(
        [sys.executable, "-m", "uvicorn", "benchmarks.mock_upstream:app",
         "--port", str(MOCK_PORT), "--no-access-log"],
        env, out / "mock.log",
    )  # fmt: skip
    gateway = start(
        [sys.executable, "-m", "uvicorn", "llm_gateway.app:app",
         "--port", str(GATEWAY_PORT), "--no-access-log"],
        env, out / "gateway.log",
    )  # fmt: skip
    try:
        wait_for(f"http://127.0.0.1:{MOCK_PORT}/v1/chat/completions")
        wait_for(f"http://127.0.0.1:{GATEWAY_PORT}/health")
        # Put the question in both caches so the hit scenarios really hit
        question = {
            "model": "bench",
            "temperature": 0,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        warm_up([question])

        mock_host = f"http://127.0.0.1:{MOCK_PORT}"
        gateway_host = f"http://127.0.0.1:{GATEWAY_PORT}"
        scenarios = [
            ("direct", mock_host),
            ("proxy", gateway_host),
            ("miss", gateway_host),
            ("exact_hit", gateway_host),
            ("semantic_hit", gateway_host),
        ]
        latency = {}
        throughput = {}
        for name, host in scenarios:
            # One request at a time: pure latency, no queueing
            latency[name] = measure_latency(name, host)
            print(f"latency    {name:13s} p50 {latency[name]['p50']:6.2f} ms", flush=True)
        for name, host in scenarios:
            # Many users at once: how many requests per second one process can handle
            throughput[name] = locust(name, THROUGHPUT_USERS, THROUGHPUT_SECONDS, host, out)
            print(f"throughput {name:13s} {throughput[name]['rps']:7.0f} req/s", flush=True)
        # Fresh Redis so hit rate and usage cover only the workload
        redis.Redis.from_url("redis://localhost:6379/2").flushdb()
        workload = run_cache_workload(gateway_host, API_KEY)
        print(
            f"workload   {workload['exact_hits']} exact + {workload['semantic_hits']} semantic "
            f"hits / {workload['requests']}, {workload['wrong_answers']} wrong",
            flush=True,
        )
    finally:
        gateway.terminate()
        mock.terminate()

    write_results(latency, throughput, workload)


def write_results(latency: dict, throughput: dict, workload: dict) -> None:
    base = latency["direct"]["p50"]
    labels = {
        "direct": "Mock upstream, called directly (baseline)",
        "proxy": "Through gateway, cache skipped",
        "miss": "Through gateway, cache miss (exact + semantic lookup, embed, store)",
        "exact_hit": "Through gateway, exact cache hit",
        "semantic_hit": "Through gateway, semantic cache hit",
    }
    lines = [
        "# Benchmark results",
        "",
        f"Machine: {platform.machine()}, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}. One gateway process (uvicorn, 1 worker), "
        "Redis 8 on localhost, mock upstream that answers instantly.",
        "",
        f"## Latency (one request at a time, {LATENCY_REQUESTS} requests per scenario)",
        "",
        "Measured from a Python client (httpx), so every row includes the same client and",
        "loopback-network cost; the baseline row shows that cost on its own.",
        "",
        "| Scenario | p50 | p95 | p99 | Added vs baseline (p50) | Failures |",
        "|---|---|---|---|---|---|",
    ]
    for name, stats in latency.items():
        added = "-" if name == "direct" else f"{stats['p50'] - base:+.2f} ms"
        lines.append(
            f"| {labels[name]} | {stats['p50']:.2f} ms | {stats['p95']:.2f} ms | "
            f"{stats['p99']:.2f} ms | {added} | {int(stats['failures'])} |"
        )
    lines += [
        "",
        f"## Throughput ({THROUGHPUT_USERS} concurrent users, {THROUGHPUT_SECONDS}s per scenario)",
        "",
        "Locust with one load-generating process. The baseline row is limited by Locust",
        "itself, not the mock; failures are mostly connections cut when each timed run stops.",
        "",
        "| Scenario | Requests/s | p50 | p95 | Failures |",
        "|---|---|---|---|---|",
    ]
    for name, stats in throughput.items():
        lines.append(
            f"| {labels[name]} | {stats['rps']:.0f} | {stats['p50']:.0f} ms | "
            f"{stats['p95']:.0f} ms | {int(stats['failures'])} |"
        )
    w = workload
    hits = w["exact_hits"] + w["semantic_hits"]
    lines += [
        "",
        "## Cache hit rate and cost (benchmarks/cache_workload.py)",
        "",
        f"{w['requests']} requests built from the 90 labeled pairs in eval/: new questions, exact",
        "repeats, rewordings, and look-alike questions that need a different answer. Priced at",
        "$3 / $15 per million input / output tokens, 50 + 100 tokens per request.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Exact cache hits | {w['exact_hits']} ({w['exact_hits'] / w['requests']:.0%}) |",
        f"| Semantic cache hits | {w['semantic_hits']} "
        f"({w['semantic_hits'] / w['requests']:.0%}) |",
        f"| Total hit rate | {hits / w['requests']:.0%} |",
        f"| Wrong cached answers | {w['wrong_answers']} |",
        f"| Cost without cache | ${w['cost_without_cache']:.4f} |",
        f"| Cost with cache | ${w['cost_with_cache']:.4f} |",
        f"| Saved | {w['saved_fraction']:.0%} |",
    ]
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- Targets: gateway overhead on a cache miss under 10 ms p50, cache hit under 20 ms p50.",
        "- Most of the cache-miss overhead is the embedding (about 2 ms for a short question, more",
        "  for longer ones; questions over `max_question_chars` skip the semantic cache).",
        "- Throughput is for a single Python process. Misses and semantic hits are limited by",
        "  CPU-bound embedding; the pass-through row by Python request handling. More uvicorn",
        "  workers or replicas scale this out (each worker loads its own copy of the model).",
        "- With a real model, upstream latency (hundreds of ms to seconds) dwarfs the gateway's",
        "  few ms, and every cache hit removes that latency entirely.",
        "",
        "Reproduce: `uv run python benchmarks/run.py` (needs Redis on localhost:6379).",
    ]
    path = ROOT / "benchmarks" / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
