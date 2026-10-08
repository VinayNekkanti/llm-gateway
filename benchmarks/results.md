# Benchmark results

Machine: arm64, Darwin 25.4.0, Python 3.12.13. One gateway process (uvicorn, 1 worker), Redis 8 on localhost, mock upstream that answers instantly.

## Latency (one request at a time, 2000 requests per scenario)

Measured from a Python client (httpx), so every row includes the same client and
loopback-network cost; the baseline row shows that cost on its own.

| Scenario | p50 | p95 | p99 | Added vs baseline (p50) | Failures |
|---|---|---|---|---|---|
| Mock upstream, called directly (baseline) | 0.19 ms | 0.23 ms | 0.26 ms | - | 0 |
| Through gateway, cache skipped | 1.27 ms | 1.55 ms | 2.00 ms | +1.07 ms | 0 |
| Through gateway, cache miss (exact + semantic lookup, embed, store) | 8.02 ms | 9.17 ms | 10.76 ms | +7.82 ms | 0 |
| Through gateway, exact cache hit | 0.88 ms | 0.96 ms | 1.00 ms | +0.69 ms | 0 |
| Through gateway, semantic cache hit | 6.26 ms | 7.51 ms | 10.55 ms | +6.07 ms | 0 |

## Throughput (50 concurrent users, 20s per scenario)

Locust with one load-generating process. The baseline row is limited by Locust
itself, not the mock; failures are mostly connections cut when each timed run stops.

| Scenario | Requests/s | p50 | p95 | Failures |
|---|---|---|---|---|
| Mock upstream, called directly (baseline) | 23847 | 2 ms | 2 ms | 0 |
| Through gateway, cache skipped | 941 | 46 ms | 100 ms | 0 |
| Through gateway, cache miss (exact + semantic lookup, embed, store) | 296 | 150 ms | 270 ms | 0 |
| Through gateway, exact cache hit | 2904 | 15 ms | 29 ms | 0 |
| Through gateway, semantic cache hit | 524 | 96 ms | 130 ms | 0 |

## Cache hit rate and cost (benchmarks/cache_workload.py)

500 requests built from the 90 labeled pairs in eval/: new questions, exact
repeats, rewordings, and look-alike questions that need a different answer. Priced at
$3 / $15 per million input / output tokens, 50 + 100 tokens per request.

| Metric | Value |
|---|---|
| Exact cache hits | 213 (43%) |
| Semantic cache hits | 96 (19%) |
| Total hit rate | 62% |
| Wrong cached answers | 0 |
| Cost without cache | $0.8250 |
| Cost with cache | $0.3151 |
| Saved | 62% |

## Reading these numbers

- Targets: gateway overhead on a cache miss under 10 ms p50, cache hit under 20 ms p50.
- Most of the cache-miss overhead is the embedding (about 2 ms for a short question, more
  for longer ones; questions over `max_question_chars` skip the semantic cache).
- Throughput is for a single Python process. Misses and semantic hits are limited by
  CPU-bound embedding; the pass-through row by Python request handling. More uvicorn
  workers or replicas scale this out (each worker loads its own copy of the model).
- With a real model, upstream latency (hundreds of ms to seconds) dwarfs the gateway's
  few ms, and every cache hit removes that latency entirely.

Reproduce: `uv run python benchmarks/run.py` (needs Redis on localhost:6379).
