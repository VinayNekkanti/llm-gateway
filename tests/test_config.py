from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import load_config

VALID = """
providers:
  ollama:
    type: openai
    base_url: http://example.test/v1
models:
  fast:
    provider: ollama
    model: llama3.2:1b
"""


def write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    monkeypatch.setenv("GATEWAY_CONFIG", str(path))


def test_loads_file_named_in_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, monkeypatch, VALID)

    config = load_config()

    assert config.providers["ollama"].base_url == "http://example.test/v1"
    assert config.providers["ollama"].timeout_seconds == 60  # default
    assert config.models["fast"].model == "llama3.2:1b"


def test_missing_required_field_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, monkeypatch, VALID.replace("    base_url: http://example.test/v1\n", ""))
    with pytest.raises(ValidationError):
        load_config()


def test_unknown_provider_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, monkeypatch, VALID.replace("provider: ollama", "provider: nope"))
    with pytest.raises(ValidationError, match="unknown provider"):
        load_config()


def test_repo_config_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    # The config.yaml we ship must always load
    monkeypatch.delenv("GATEWAY_CONFIG", raising=False)
    assert load_config().models
