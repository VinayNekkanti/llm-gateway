from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.auth import load_api_keys, require_api_key
from llm_gateway.config import Config, load_config
from llm_gateway.errors import GatewayError, gateway_error_handler


def create_app(config: Config | None = None, api_keys: list[str] | None = None) -> FastAPI:
    """Build the gateway. Tests pass their own config and keys; normal runs load them at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once when the server starts
        app.state.config = config or load_config()
        app.state.api_keys = api_keys or load_api_keys()
        # One shared HTTP client: reuses connections to the LLM instead of opening one per request
        async with httpx.AsyncClient(timeout=app.state.config.upstream.timeout_seconds) as http:
            app.state.http = http
            yield
        # Leaving the "async with" closes the client when the server stops

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.add_exception_handler(GatewayError, gateway_error_handler)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, key: str = Depends(require_api_key)) -> Response:
        # 1. Read the request the app sent us
        body = await request.json()
        http: httpx.AsyncClient = request.app.state.http
        url: str = request.app.state.config.upstream.url

        # 2a. Streaming: send pieces back as they arrive
        if body.get("stream"):
            return StreamingResponse(
                stream_from_upstream(http, url, body),
                media_type="text/event-stream",
            )

        # 2b. Not streaming: wait for the full answer, then send it back
        upstream = await http.post(url, json=body)
        return JSONResponse(content=upstream.json(), status_code=upstream.status_code)

    return app


async def stream_from_upstream(
    http: httpx.AsyncClient, url: str, body: dict
) -> AsyncIterator[bytes]:
    # Open a streaming connection to the LLM and pass along each piece as it arrives
    async with http.stream("POST", url, json=body) as upstream:
        async for chunk in upstream.aiter_bytes():
            yield chunk


# uvicorn llm_gateway.app:app looks for this
app = create_app()
