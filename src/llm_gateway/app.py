import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Where the gateway sends requests. Ollama speaks the same format as OpenAI.
UPSTREAM_URL = "http://localhost:11434/v1/chat/completions"

app = FastAPI(title="LLM Gateway")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    # 1. Read the request the app sent us
    body = await request.json()

    # 2. Forward it to the LLM and wait for the answer
    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream = await client.post(UPSTREAM_URL, json=body)

    # 3. Send the LLM's answer back to the app
    return JSONResponse(content=upstream.json(), status_code=upstream.status_code)