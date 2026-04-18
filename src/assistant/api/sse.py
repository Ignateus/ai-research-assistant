"""Server-Sent Event helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator


def sse_event(data: dict | str, event: str | None = None) -> str:
    """
    Format a single SSE message.

    Args:
        data:  Either a dict (serialised to JSON) or a plain string.
        event: Optional SSE event name (e.g. "chunk", "done", "error").

    Returns:
        A formatted SSE string ready to be yielded from a StreamingResponse.
    """
    payload = json.dumps(data) if isinstance(data, dict) else data
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    lines.append("")   # blank line terminates the event
    lines.append("")
    return "\n".join(lines)


async def aiter_sse(sync_gen) -> AsyncGenerator[str, None]:  # noqa: ANN001
    """
    Wrap a synchronous generator in an async generator for use with StreamingResponse.

    FastAPI's StreamingResponse accepts async generators; this bridges the gap
    when the underlying source (e.g. Anthropic SDK streaming) is synchronous.
    """
    for item in sync_gen:
        yield item
