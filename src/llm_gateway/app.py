from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# Where the gateway sends requests. Ollama speaks the same format as OpenAI.
UPSTREAM_URL = "http://localhost:11434/v1/chat/completions"

app = FastAPI(title="LLM Gateway")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def stream_from_upstream(body: dict) -> AsyncIterator[bytes]:
    # Open a streaming connection to the LLM and pass along each piece as it arrives
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", UPSTREAM_URL, json=body) as upstream:
            async for chunk in upstream.aiter_bytes():
                yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    # 1. Read the request the app sent us
    body = await request.json()

    # 2a. Streaming (like the chatbot): send pieces back as they arrive
    if body.get("stream"):
        return StreamingResponse(
            stream_from_upstream(body),
            media_type="text/event-stream",
        )

    # 2b. Not streaming (like the email sorter): wait for the full answer
    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream = await client.post(UPSTREAM_URL, json=body)
    return JSONResponse(content=upstream.json(), status_code=upstream.status_code)
