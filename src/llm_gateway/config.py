import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class UpstreamConfig(BaseModel):
    url: str
    timeout_seconds: float = 60.0


class Config(BaseModel):
    upstream: UpstreamConfig


def load_config() -> Config:
    # Use the file named in GATEWAY_CONFIG if set, otherwise config.yaml
    path = Path(os.environ.get("GATEWAY_CONFIG", "config.yaml"))
    with path.open() as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
