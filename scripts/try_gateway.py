from openai import OpenAI

# Point the official OpenAI client at YOUR gateway instead of OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed-yet")

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "What is an LLM gateway in one sentence?"}],
)

print(response.choices[0].message.content)
