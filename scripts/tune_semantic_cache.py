"""Pick the semantic cache threshold from labeled question pairs.

Each pair in eval/semantic_cache_pairs.jsonl is labeled same=true (should share an answer)
or same=false (needs a different answer). For each threshold we count:
  - hits on same pairs       (good: saved a model call)
  - false hits on different  (bad: a user gets the answer to someone else's question)
A false hit is much worse than a miss, so we pick the lowest threshold with zero false hits.

Run: uv run python scripts/tune_semantic_cache.py [pairs file]
Default pairs file is the tuning set; pass eval/semantic_cache_pairs_holdout.jsonl to check
the chosen threshold on pairs that weren't used to design the text checks.
"""

import json
import sys
from pathlib import Path

import numpy as np

from llm_gateway.cache.semantic import FastEmbedEmbedder, likely_different_question

DEFAULT = Path(__file__).parent.parent / "eval" / "semantic_cache_pairs.jsonl"
PAIRS = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT

pairs = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")


def cosine(a: str, b: str) -> float:
    va, vb = np.array(embedder.embed(a)), np.array(embedder.embed(b))
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


scored = [(cosine(p["a"], p["b"]), p) for p in pairs]
# The text checks run after the similarity check; a blocked pair can never hit
blocked = {id(p) for _, p in scored if likely_different_question(p["a"], p["b"])}
same = sorted(s for s, p in scored if p["same"])
different = sorted(s for s, p in scored if not p["same"])

print(f"{len(same)} same-meaning pairs:  min {same[0]:.3f}  median {np.median(same):.3f}")
print(f"{len(different)} different pairs:     max {different[-1]:.3f}", end="")
print(f"  median {np.median(different):.3f}")
print()
print("              embeddings only          + text checks")
print("threshold   hits (same)  false hits   hits (same)  false hits")
best = None
for threshold in np.arange(0.80, 0.991, 0.01):
    hits = sum(s >= threshold for s, p in scored if p["same"])
    false_hits = sum(s >= threshold for s, p in scored if not p["same"])
    g_hits = sum(s >= threshold and id(p) not in blocked for s, p in scored if p["same"])
    g_false = sum(s >= threshold and id(p) not in blocked for s, p in scored if not p["same"])
    print(
        f"  {threshold:.2f}      {hits:2d}/{len(same)}        {false_hits:2d}"
        f"           {g_hits:2d}/{len(same)}        {g_false:2d}"
    )
    if best is None and g_false == 0:
        best = (threshold, g_hits)

print()
print("Hardest different pairs that pass the text checks (closest to a false hit):")
unblocked = [(s, p) for s, p in scored if not p["same"] and id(p) not in blocked]
for s, p in sorted(unblocked, key=lambda x: -x[0])[:5]:
    print(f"  {s:.3f}  {p['a']!r} vs {p['b']!r}")
print("Same pairs wrongly blocked by the text checks:")
for s, p in scored:
    if p["same"] and id(p) in blocked:
        print(f"  {s:.3f}  {p['a']!r} vs {p['b']!r}")
print("Hardest same pairs (closest to a miss):")
for s, p in sorted(((s, p) for s, p in scored if p["same"]), key=lambda x: x[0])[:5]:
    print(f"  {s:.3f}  {p['a']!r} vs {p['b']!r}")

if best:
    print(
        f"\nLowest threshold with zero false hits (with text checks): {best[0]:.2f} "
        f"({best[1]}/{len(same)} hits)"
    )
