import os
import re
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


class Pricing(BaseModel):
    # US dollars per million tokens, from the provider's price list
    input_per_million: float = 0.0
    output_per_million: float = 0.0


class ModelConfig(BaseModel):
    provider: str
    # The model name the provider expects, e.g. "llama3.2:1b"
    model: str
    # Request fields to remove before sending, for providers that reject them
    drop_params: list[str] = []
    # Other model names to try, in order, if this one keeps failing
    fallbacks: list[str] = []
    # Used for cost estimates; leave out for free/local models
    pricing: Pricing | None = None


class RetryConfig(BaseModel):
    # Total tries per model (1 = no retries)
    max_attempts: int = 3
    # Wait before the first retry; doubles each time (plus a little randomness), up to the max
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    # Keep this short: a slow Redis must never make the gateway slow
    timeout_seconds: float = 0.25


class ExactCacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600


class SemanticCacheConfig(BaseModel):
    enabled: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the embedding model is downloaded to (None = fastembed's default temp folder)
    model_cache_dir: str | None = None
    # Minimum cosine similarity to reuse an answer. Tuned with scripts/tune_semantic_cache.py
    similarity_threshold: float = 0.94
    ttl_seconds: int = 3600
    # Embedding time grows with length (~2 ms short question, ~9 ms for 100 words), and long
    # prompts rarely get reworded matches, so longer questions skip the semantic cache
    max_question_chars: int = 300


class CacheConfig(BaseModel):
    # False: each API key has its own cache entries, so one client never sees another's answers
    shared_across_keys: bool = False
    # Only cache requests at or below this temperature (missing temperature counts as 1.0)
    max_temperature: float = 0.3
    exact: ExactCacheConfig = ExactCacheConfig()
    semantic: SemanticCacheConfig = SemanticCacheConfig()


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = 60
    window_seconds: int = 60
    # Different limits for specific keys, by key id (the "key_..." name shown in logs)
    per_key: dict[str, int] = {}


class Config(BaseModel):
    providers: dict[str, ProviderConfig]
    # Model names clients can send -> where each one goes
    models: dict[str, ModelConfig]
    retry: RetryConfig = RetryConfig()
    redis: RedisConfig = RedisConfig()
    cache: CacheConfig = CacheConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()

    @model_validator(mode="after")
    def check_references(self) -> "Config":
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"Model '{name}' uses unknown provider '{model.provider}'")
            for fallback in model.fallbacks:
                if fallback not in self.models:
                    raise ValueError(f"Model '{name}' has unknown fallback '{fallback}'")
        return self


# ${NAME} or ${NAME:-default}
ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Replace ${NAME:-default} with the environment variable NAME (or the default).

    Lets one config.yaml work locally and in Docker, where addresses differ.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise ValueError(f"config.yaml uses ${{{name}}} but it is not set")

    return ENV_VAR.sub(replace, text)


def load_config() -> Config:
    # Use the file named in GATEWAY_CONFIG if set, otherwise config.yaml
    path = Path(os.environ.get("GATEWAY_CONFIG", "config.yaml"))
    data = yaml.safe_load(expand_env(path.read_text()))
    return Config.model_validate(data)
