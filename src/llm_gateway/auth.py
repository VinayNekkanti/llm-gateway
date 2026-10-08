import hashlib
import os
import secrets

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_gateway.errors import GatewayError

# Reads "Authorization: Bearer <key>". auto_error=False so we can send our own error body.
bearer = HTTPBearer(auto_error=False)


def load_api_keys() -> list[str]:
    """Read allowed keys from GATEWAY_API_KEYS (comma-separated)."""
    raw = os.environ.get("GATEWAY_API_KEYS", "")
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        # Fail fast: a gateway with no keys would either reject everything or be wide open
        raise RuntimeError(
            "GATEWAY_API_KEYS is not set. Add it to .env and start with --env-file .env"
        )
    return keys


def key_id(api_key: str) -> str:
    """A short, safe name for a key, used in logs and usage stats instead of the key itself."""
    return "key_" + hashlib.sha256(api_key.encode()).hexdigest()[:12]


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    """FastAPI dependency: reject the request unless it has a valid key. Returns the key id."""
    if credentials is None:
        raise GatewayError(
            401,
            "Missing API key. Send it as 'Authorization: Bearer <key>'.",
            "invalid_request_error",
            "missing_api_key",
        )

    sent = credentials.credentials.encode()
    # compare_digest takes the same time whether the keys match or not (no timing attacks)
    for allowed in request.app.state.api_keys:
        if secrets.compare_digest(sent, allowed.encode()):
            return key_id(allowed)

    raise GatewayError(401, "Invalid API key.", "invalid_request_error", "invalid_api_key")
