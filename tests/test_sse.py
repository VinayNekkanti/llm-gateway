from collections.abc import AsyncIterator

from llm_gateway.sse import iter_sse_events, sse_data


async def from_chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(*chunks: bytes) -> list[tuple[str | None, str]]:
    return [event async for event in iter_sse_events(from_chunks(*chunks))]


async def test_events_split_across_chunks_are_joined() -> None:
    events = await collect(b"event: a\nda", b"ta: 1\n", b"\ndata: 2\n\n")
    assert events == [("a", "1"), (None, "2")]


async def test_windows_line_endings() -> None:
    assert await collect(b"data: x\r\n\r\n") == [(None, "x")]


async def test_comment_only_events_are_skipped() -> None:
    assert await collect(b": keep-alive\n\ndata: y\n\n") == [(None, "y")]


def test_sse_data_encodes_json_and_strings() -> None:
    assert sse_data({"a": 1}) == b'data: {"a": 1}\n\n'
    assert sse_data("[DONE]") == b"data: [DONE]\n\n"
