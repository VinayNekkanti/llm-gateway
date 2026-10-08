from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.config import load_config

# Read settings from config.yaml once, when the gateway starts
config = load_config()

app = FastAPI(title="LLM Gateway")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def stream_from_upstream(body: dict) -> AsyncIterator[bytes]:
    # Open a streaming connection to the LLM and pass along each piece as it arrives
    async with httpx.AsyncClient(timeout=config.upstream.timeout_seconds) as client:
        async with client.stream("POST", config.upstream.url, json=body) as upstream:
            async for chunk in upstream.aiter_bytes():
                yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    # 1. Read the request the app sent us
    body = await request.json()

    # 2a. Streaming: send pieces back as they arrive
    if body.get("stream"):
        return StreamingResponse(
            stream_from_upstream(body),
            media_type="text/event-stream",
        )

    # 2b. Not streaming: wait for the full answer, then send it back
    async with httpx.AsyncClient(timeout=config.upstream.timeout_seconds) as client:
        upstream = await client.post(config.upstream.url, json=body)
    return JSONResponse(content=upstream.json(), status_code=upstream.status_code)
