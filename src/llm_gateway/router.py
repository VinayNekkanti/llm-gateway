import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_gateway.config import Config, ModelConfig
from llm_gateway.errors import GatewayError
from llm_gateway.observability import FALLBACKS, UPSTREAM_ATTEMPTS, UPSTREAM_LATENCY
from llm_gateway.providers import Provider, ProviderError

T = TypeVar("T")
logger = structlog.get_logger()


@dataclass
class Target:
    """One place a request can go: a provider plus the model name that provider expects."""

    name: str  # the model name from config.yaml
    provider: Provider
    model: ModelConfig

    def prepare(self, body: dict[str, Any]) -> dict[str, Any]:
        """Copy the client's request, swap in the provider's model name, drop unsupported fields."""
        upstream = {k: v for k, v in body.items() if k not in self.model.drop_params}
        upstream["model"] = self.model.model
        return upstream


async def _measure[R](call: Callable[[Target], Awaitable[R]], target: Target) -> R:
    provider = target.provider.name
    start = time.perf_counter()
    try:
        result = await call(target)
    except ProviderError as exc:
        UPSTREAM_ATTEMPTS.labels(provider, target.name, str(exc.status_code)).inc()
        raise
    finally:
        UPSTREAM_LATENCY.labels(provider).observe(time.perf_counter() - start)
    UPSTREAM_ATTEMPTS.labels(provider, target.name, "ok").inc()
    return result


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


class Router:
    def __init__(self, config: Config, providers: dict[str, Provider]) -> None:
        self.config = config
        self.providers = providers

    def resolve(self, model_name: str) -> Target:
        model = self.config.models.get(model_name)
        if model is None:
            raise GatewayError(
                404,
                f"The model '{model_name}' does not exist. "
                f"Available: {', '.join(sorted(self.config.models))}",
                "invalid_request_error",
                "model_not_found",
            )
        return Target(model_name, self.providers[model.provider], model)

    def targets(self, model_name: str) -> list[Target]:
        """The requested model first, then its fallbacks in order."""
        first = self.resolve(model_name)
        return [first] + [self.resolve(name) for name in first.model.fallbacks]

    async def chat(self, body: dict[str, Any]) -> tuple[dict[str, Any], Target]:
        return await self._with_fallback(
            body["model"], lambda target: target.provider.chat(target.prepare(body))
        )

    async def open_stream(self, body: dict[str, Any]) -> tuple[AsyncIterator[bytes], Target]:
        # Retries and fallback only happen before the first byte. Once text is flowing
        # to the client we can't take it back, so a mid-stream failure just ends the stream.
        return await self._with_fallback(
            body["model"], lambda target: target.provider.open_stream(target.prepare(body))
        )

    async def _with_fallback(
        self, model_name: str, call: Callable[[Target], Awaitable[T]]
    ) -> tuple[T, Target]:
        last_error: ProviderError | None = None
        targets = self.targets(model_name)
        for index, target in enumerate(targets):
            try:
                return await self._with_retries(call, target), target
            except ProviderError as exc:
                if not exc.retryable:
                    # The request itself is bad (e.g. 400): another model won't fix it
                    raise
                last_error = exc  # this model is down or overloaded: try the next one
                if index + 1 < len(targets):
                    FALLBACKS.labels(target.name, targets[index + 1].name).inc()
                    logger.warning(
                        "falling_back",
                        from_model=target.name,
                        to_model=targets[index + 1].name,
                        status=exc.status_code,
                    )
        assert last_error is not None
        raise last_error

    async def _with_retries(self, call: Callable[[Target], Awaitable[T]], target: Target) -> T:
        retry = self.config.retry
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry.max_attempts),
            # Exponential backoff with jitter: 0.5s, 1s, 2s... plus randomness so many
            # clients don't all retry at the same instant
            wait=wait_exponential_jitter(
                multiplier=retry.initial_backoff_seconds,
                max=retry.max_backoff_seconds,
                jitter=retry.initial_backoff_seconds,
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return await _measure(call, target)
        raise AssertionError("unreachable")  # tenacity either returns or raises
