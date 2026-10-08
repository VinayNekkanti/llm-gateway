# Semantic cache: design and threshold tuning

The exact cache only helps when a request is byte-for-byte identical. The semantic cache
reuses an answer when a new question *means* the same as one already answered:
"What is the capital of France?" and "What's France's capital city?" share an answer.

## How a lookup works

1. The request is split into the **question** (the last user message) and the **context**
   (system prompt, earlier turns, parameters, the real provider/model, and the API key
   unless `shared_across_keys` is on).
2. The context must match exactly, so it is hashed. "Summarize this" with a different
   document attached is a different request, no matter how similar the words are.
3. The question is embedded locally with `BAAI/bge-small-en-v1.5` (384 dimensions,
   about 2 ms on a laptop CPU, run in a worker thread so it never blocks the event loop).
4. Redis 8 **vector sets** (`VADD` / `VSIM`) find the nearest stored question *with the same
   context hash* (`FILTER '.ctx == "<hash>"'`).
5. The match is used only if its cosine similarity is at least the threshold **and** it
   passes the text checks below.

Answers are stored under their own key with a TTL. When an answer expires its vector is
left behind; the next lookup that finds it deletes it (lazy cleanup).

## Why a threshold alone is not safe

Tuned on 60 labeled pairs (`eval/semantic_cache_pairs.jsonl`): 30 paraphrases that should
share an answer and 30 "hard negatives" that look alike but need different answers.

Embeddings mostly capture *which words* appear, not their order or small edits:

| Pair | Cosine | Same answer? |
|---|---|---|
| "Convert 10 miles to kilometers." / "Convert 10 kilometers to miles." | 0.992 | No |
| "...convert a string to an integer..." / "...convert an integer to a string..." | 0.991 | No |
| "Should I use tabs...?" / "Should I not use tabs...?" | 0.966 | No |
| "What is the speed of light?" / "How fast does light travel?" | 0.842 | Yes |

The worst false match scores *higher* than every real paraphrase, so no threshold works on
its own. At 0.95 (a common default), embeddings alone hit only 53% of paraphrases and still
return the wrong answer for 4 of 30 different questions.

## Text checks

Cheap rules that catch the edits embeddings miss (`likely_different_question`):

- both questions contain numbers, and the numbers differ
- one question is negated and the other isn't
- same words in a different order
- same length with exactly one word swapped ("France" vs "Germany")

## Results

Tuning set (60 pairs; the text checks were designed while looking at it):

| Threshold | Paraphrases hit, embeddings only | Wrong answers, embeddings only | Paraphrases hit, + text checks | Wrong answers, + text checks |
|---|---|---|---|---|
| 0.84 | 30/30 | 11 | 30/30 | 0 |
| 0.90 | 26/30 | 5 | 26/30 | 0 |
| 0.94 | 16/30 | 4 | 16/30 | 0 |

Because the rules were written while looking at the tuning set, those numbers are optimistic.
A **held-out set** of 30 new pairs (`eval/semantic_cache_pairs_holdout.jsonl`, written after
the rules were fixed) showed it:

| Threshold | Paraphrases hit | Wrong answers |
|---|---|---|
| 0.84 | 14/15 | 2 |
| 0.90 | 11/15 | 1 |
| 0.94 | 6/15 | 0 |

The held-out failure at 0.84–0.93 was "length of a **string**" vs "length of an **array**"
(two words differ, so the one-word-swap rule misses it).

**Chosen threshold: 0.94.** Across all 90 pairs: 22 of 45 paraphrases served from cache
(49%), 0 of 45 wrong answers. A miss costs one model call; a wrong answer costs the user's
trust, so the threshold is set for zero wrong answers rather than maximum hit rate.

Reproduce:

```bash
uv run python scripts/tune_semantic_cache.py
uv run python scripts/tune_semantic_cache.py eval/semantic_cache_pairs_holdout.jsonl
```

## Limits and next steps

- 90 pairs is a small evaluation; real traffic should be sampled and labeled before
  lowering the threshold.
- A cross-encoder or a small LLM judge could verify candidate matches and allow a lower
  threshold (higher hit rate) at the cost of extra latency on near matches.
- Only the last user message is embedded; multi-turn conversations match only when all
  earlier turns are identical.
- Streaming requests are not cached.
