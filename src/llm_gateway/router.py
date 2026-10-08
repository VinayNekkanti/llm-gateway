from dataclasses import dataclass
from typing import Any

from llm_gateway.config import Config, ModelConfig
from llm_gateway.errors import GatewayError
from llm_gateway.providers import Provider


@dataclass
class Target:
    """One place a request can go: a provider plus the model name that provider expects."""

    name: str  # the model name the client used
    provider: Provider
    model: ModelConfig

    def prepare(self, body: dict[str, Any]) -> dict[str, Any]:
        """Copy the client's request, swap in the provider's model name, drop unsupported fields."""
        upstream = {k: v for k, v in body.items() if k not in self.model.drop_params}
        upstream["model"] = self.model.model
        return upstream


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
