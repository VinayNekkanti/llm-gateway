import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.auth import load_api_keys, require_api_key
from llm_gateway.config import Config, load_config
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.providers import ProviderError, build_providers
from llm_gateway.router import Router, Target


def create_app(config: Config | None = None, api_keys: list[str] | None = None) -> FastAPI:
    """Build the gateway. Tests pass their own config and keys; normal runs load them at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once when the server starts
        app.state.config = config or load_config()
        app.state.api_keys = api_keys or load_api_keys()
        # One shared HTTP client: reuses connections to the LLM instead of opening one per request
        async with httpx.AsyncClient() as http:
            providers = build_providers(app.state.config, http)
            app.state.router = Router(app.state.config, providers)
            yield
        # Leaving the "async with" closes the client when the server stops

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(ProviderError, provider_error_handler)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request, key: str = Depends(require_api_key)) -> dict[str, Any]:
        # Same shape as OpenAI's model list, so SDKs and tools can discover our model names
        names = sorted(request.app.state.config.models)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "llm-gateway"} for name in names],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, key: str = Depends(require_api_key)) -> Response:
        # 1. Read and check the request the app sent us
        body = await read_chat_request(request)

        # 2. Send it to the requested model, with retries and fallback models
        router: Router = request.app.state.router

        # 3a. Streaming: send pieces back as they arrive
        if body.get("stream"):
            chunks, target = await router.open_stream(body)
            return StreamingResponse(
                chunks, media_type="text/event-stream", headers=served_by(target)
            )

        # 3b. Not streaming: wait for the full answer, then send it back
        result, target = await router.chat(body)
        return JSONResponse(result, headers=served_by(target))

    return app


async def read_chat_request(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise GatewayError(
            400, "Request body must be valid JSON.", "invalid_request_error"
        ) from exc
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise GatewayError(400, "'model' is required.", "invalid_request_error")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise GatewayError(400, "'messages' must be a non-empty list.", "invalid_request_error")
    return body


def served_by(target: Target) -> dict[str, str]:
    # Tells the client which model actually answered (differs from the request after a fallback)
    return {"x-gateway-model": target.name}


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)
    return JSONResponse(status_code=exc.status_code, content=exc.body)


# uvicorn llm_gateway.app:app looks for this
app = create_app()
