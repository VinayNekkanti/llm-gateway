import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_gateway.auth import load_api_keys, require_api_key
from llm_gateway.cache.exact import ExactCache
from llm_gateway.cache.semantic import Embedder, FastEmbedEmbedder, SemanticCache
from llm_gateway.config import Config, load_config
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.gateway import Gateway
from llm_gateway.observability import HTTP_LATENCY, HTTP_REQUESTS, setup_logging
from llm_gateway.providers import ProviderError, build_providers
from llm_gateway.ratelimit import RateLimiter, enforce_rate_limit
from llm_gateway.redis_store import RedisStore
from llm_gateway.router import Router
from llm_gateway.usage import UsageTracker

logger = structlog.get_logger()
CallNext = Callable[[Request], Awaitable[Response]]


def create_app(
    config: Config | None = None,
    api_keys: list[str] | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the gateway. Tests pass their own config and keys; normal runs load them at startup."""

    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once when the server starts
        app.state.config = config or load_config()
        app.state.api_keys = api_keys or load_api_keys()
        cfg: Config = app.state.config
        store = RedisStore(cfg.redis.url, cfg.redis.timeout_seconds)
        # One shared HTTP client: reuses connections to the LLM instead of opening one per request
        async with httpx.AsyncClient() as http:
            router = Router(cfg, build_providers(cfg, http))
            exact_cache = (
                ExactCache(store, cfg.cache.exact.ttl_seconds) if cfg.cache.exact.enabled else None
            )
            app.state.usage = UsageTracker(store)
            app.state.gateway = Gateway(
                cfg,
                router,
                exact_cache,
                build_semantic_cache(cfg, store, embedder),
                app.state.usage,
            )
            app.state.rate_limiter = (
                RateLimiter(store, cfg.rate_limit) if cfg.rate_limit.enabled else None
            )
            yield
        # Server is stopping: the "async with" closed the HTTP client; now close Redis
        await store.close()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(ProviderError, provider_error_handler)

    @app.middleware("http")
    async def observe(request: Request, call_next: CallNext) -> Response:
        """Request ID, one log line and metrics for every request."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled_error", path=request.url.path)
            raise
        duration = time.perf_counter() - start

        # Route template ("/v1/chat/completions"), not the raw path: keeps metric labels bounded
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(route_path, request.method, str(response.status_code)).inc()
        HTTP_LATENCY.labels(route_path).observe(duration)
        response.headers["x-request-id"] = request_id
        if route_path not in ("/health", "/metrics"):
            logger.info(
                "request",
                method=request.method,
                route=route_path,
                status=response.status_code,
                duration_ms=round(duration * 1000, 2),
                key_id=getattr(request.state, "key_id", None),
                model=response.headers.get("x-gateway-model"),
                cache=response.headers.get("x-gateway-cache"),
            )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        # Prometheus scrapes this. Keep it on an internal network: it shows traffic volumes.
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models(request: Request, key: str = Depends(require_api_key)) -> dict[str, Any]:
        # Same shape as OpenAI's model list, so SDKs and tools can discover our model names
        names = sorted(request.app.state.config.models)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "llm-gateway"} for name in names],
        }

    @app.get("/usage")
    async def usage(
        request: Request,
        days: int = Query(30, ge=1, le=90),
        key: str = Depends(require_api_key),
    ) -> dict[str, Any]:
        """Tokens, estimated cost and cache savings for the calling API key only."""
        tracker: UsageTracker = request.app.state.usage
        report = await tracker.report(key, days)
        if report is None:
            raise GatewayError(503, "Usage data is unavailable right now.", "api_error")
        return report

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, key: str = Depends(enforce_rate_limit)
    ) -> Response:
        # 1. Read and check the request the app sent us
        body = await read_chat_request(request)

        # 2. Run it through the pipeline: cache, retries, fallback
        gateway: Gateway = request.app.state.gateway
        skip_cache = request.headers.get("x-gateway-cache", "").lower() == "skip"
        result = await gateway.chat(body, key, skip_cache=skip_cache)

        # 3. Send back either a stream or the full answer
        if result.stream is not None:
            return StreamingResponse(
                result.stream, media_type="text/event-stream", headers=result.headers
            )
        return JSONResponse(result.body, headers=result.headers)

    return app


def build_semantic_cache(
    cfg: Config, store: RedisStore, embedder: Embedder | None = None
) -> SemanticCache | None:
    semantic = cfg.cache.semantic
    if not semantic.enabled:
        return None
    # Loads the embedding model once at startup (downloads it the first time)
    embedder = embedder or FastEmbedEmbedder(semantic.embedding_model, semantic.model_cache_dir)
    return SemanticCache(
        store,
        embedder,
        semantic.similarity_threshold,
        semantic.ttl_seconds,
        semantic.max_question_chars,
    )


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


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)
    return JSONResponse(status_code=exc.status_code, content=exc.body)


# uvicorn llm_gateway.app:app looks for this
app = create_app()
