import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ["GATEWAY_API_KEY"],
)

stream = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Write a short paragraph about why caching matters."}],
    stream=True,
)

# Print each piece of text the moment it arrives
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()
