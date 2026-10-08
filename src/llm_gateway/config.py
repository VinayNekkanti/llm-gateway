import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class ProviderConfig(BaseModel):
    # "openai" = any OpenAI-compatible API (Ollama, OpenAI, vLLM); "anthropic" = Messages API
    type: Literal["openai", "anthropic"]
    base_url: str
    # Name of the environment variable holding this provider's API key (never the key itself)
    api_key_env: str | None = None
    timeout_seconds: float = 60.0


class ModelConfig(BaseModel):
    provider: str
    # The model name the provider expects, e.g. "llama3.2:1b"
    model: str
    # Request fields to remove before sending, for providers that reject them
    drop_params: list[str] = []
    # Other model names to try, in order, if this one keeps failing
    fallbacks: list[str] = []


class RetryConfig(BaseModel):
    # Total tries per model (1 = no retries)
    max_attempts: int = 3
    # Wait before the first retry; doubles each time (plus a little randomness), up to the max
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0


class Config(BaseModel):
    providers: dict[str, ProviderConfig]
    # Model names clients can send -> where each one goes
    models: dict[str, ModelConfig]
    retry: RetryConfig = RetryConfig()

    @model_validator(mode="after")
    def check_references(self) -> "Config":
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"Model '{name}' uses unknown provider '{model.provider}'")
            for fallback in model.fallbacks:
                if fallback not in self.models:
                    raise ValueError(f"Model '{name}' has unknown fallback '{fallback}'")
        return self


def load_config() -> Config:
    # Use the file named in GATEWAY_CONFIG if set, otherwise config.yaml
    path = Path(os.environ.get("GATEWAY_CONFIG", "config.yaml"))
    with path.open() as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
