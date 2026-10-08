from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import load_config


def test_loads_file_named_in_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("upstream:\n  url: http://example.test/v1/chat/completions\n")
    monkeypatch.setenv("GATEWAY_CONFIG", str(path))

    config = load_config()

    assert config.upstream.url == "http://example.test/v1/chat/completions"
    assert config.upstream.timeout_seconds == 60  # default


def test_missing_required_field_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("upstream:\n  timeout_seconds: 5\n")
    monkeypatch.setenv("GATEWAY_CONFIG", str(path))

    with pytest.raises(ValidationError):
        load_config()
