import os


def main() -> None:
    """`uv run llm-gateway`: start the server. HOST and PORT can be set in the environment."""
    import uvicorn

    uvicorn.run(
        "llm_gateway.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        # The gateway writes its own structured request log, so uvicorn's is redundant
        access_log=False,
    )
