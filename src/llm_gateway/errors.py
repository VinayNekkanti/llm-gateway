from fastapi import Request
from fastapi.responses import JSONResponse


class GatewayError(Exception):
    """An error we send back to the client in OpenAI's error format."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers


def error_body(message: str, error_type: str, code: str | None = None) -> dict:
    # Same shape OpenAI uses, so OpenAI SDKs can parse our errors
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


async def gateway_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.message, exc.error_type, exc.code),
        headers=exc.headers,
    )
