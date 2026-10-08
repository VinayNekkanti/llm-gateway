from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_gateway.config import Config, ModelConfig
from llm_gateway.errors import GatewayError
from llm_gateway.providers import Provider, ProviderError

T = TypeVar("T")


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
        for target in self.targets(model_name):
            try:
                return await self._with_retries(call, target), target
            except ProviderError as exc:
                if not exc.retryable:
                    # The request itself is bad (e.g. 400): another model won't fix it
                    raise
                last_error = exc  # this model is down or overloaded: try the next one
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
                return await call(target)
        raise AssertionError("unreachable")  # tenacity either returns or raises
