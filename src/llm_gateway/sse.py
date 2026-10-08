"""Server-Sent Events helpers: the "data: ...\\n\\n" format LLM APIs use for streaming."""

import json
from collections.abc import AsyncIterator
from typing import Any


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[tuple[str | None, str]]:
    """Turn raw bytes into (event name, data) pairs. Events can be split across network chunks."""
    buffer = b""
    async for chunk in chunks:
        buffer = (buffer + chunk).replace(b"\r\n", b"\n")
        # A blank line ends one event
        while b"\n\n" in buffer:
            raw, buffer = buffer.split(b"\n\n", 1)
            event: str | None = None
            data_lines = []
            for line in raw.decode().split("\n"):
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
            if data_lines:
                yield event, "\n".join(data_lines)


def sse_data(payload: dict[str, Any] | str) -> bytes:
    """Encode one SSE event in OpenAI's style (data only, no event name)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {text}\n\n".encode()
