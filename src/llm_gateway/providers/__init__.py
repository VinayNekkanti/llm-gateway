import httpx

from llm_gateway.config import Config
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.providers.openai_compat import OpenAICompatibleProvider

PROVIDER_TYPES: dict[str, type[Provider]] = {
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


def build_providers(config: Config, http: httpx.AsyncClient) -> dict[str, Provider]:
    """Create one provider object per entry under `providers:` in config.yaml."""
    return {
        name: PROVIDER_TYPES[provider.type](name, provider, http)
        for name, provider in config.providers.items()
    }


__all__ = ["Provider", "ProviderError", "build_providers"]
