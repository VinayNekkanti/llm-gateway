"""Send the same request twice and compare: the second should come from the cache."""

import os
import time

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ["GATEWAY_API_KEY"],
)

for attempt in (1, 2):
    start = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "Name three primary colors."}],
        temperature=0,  # low temperature: allowed to be cached
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = raw.parse().choices[0].message.content
    cache = raw.headers.get("x-gateway-cache")
    print(f"Request {attempt}: cache={cache}, {elapsed_ms:.0f} ms -> {answer[:60]!r}")
