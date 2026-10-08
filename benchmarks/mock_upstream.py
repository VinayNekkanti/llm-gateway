"""A fake OpenAI-compatible model that answers instantly, so benchmarks measure the gateway.

Run: uv run uvicorn benchmarks.mock_upstream:app --port 9100 --no-access-log
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# Fixed token counts so cost numbers are predictable
PROMPT_TOKENS = 50
COMPLETION_TOKENS = 100


async def chat(request: Request) -> JSONResponse:
    body = await request.json()
    question = body["messages"][-1]["content"]
    return JSONResponse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    # Echo the question so benchmarks can check a cached answer fits the question
                    "message": {"role": "assistant", "content": f"Answer to: {question}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }
    )


app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
