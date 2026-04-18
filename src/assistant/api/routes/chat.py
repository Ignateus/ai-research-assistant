"""Chat route — POST /chat streams assistant replies as SSE."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...client import AssistantClient, ConversationSession, Message
from ...tools.registry import ToolRegistry
from ..deps import get_client, get_registry
from ..models import ChatRequest
from ..sse import sse_event

router = APIRouter(prefix="/chat", tags=["chat"])

_SYSTEM = """\
You are an expert research assistant. Answer clearly and in depth. \
Use tools when needed to provide accurate, up-to-date information.\
"""


@router.post(
    "",
    summary="Chat with the assistant (streaming)",
    response_description="Server-Sent Events stream of text chunks",
)
async def chat(
    request: ChatRequest,
    client: AssistantClient = Depends(get_client),
    registry: ToolRegistry = Depends(get_registry),
) -> StreamingResponse:
    """
    Send a conversation to the assistant and receive a streaming reply.

    The response is a stream of Server-Sent Events:
    - `event: chunk` — a text chunk from the assistant
    - `event: done`  — signals the end of the response
    - `event: error` — an error occurred

    Pass the full conversation history in `messages` on each request.
    The API is stateless — no server-side session is stored.
    """
    system = request.system or _SYSTEM
    session = ConversationSession(system_prompt=system)

    for msg in request.messages:
        session.history.append(Message(role=msg.role, content=msg.content))

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            for chunk in client.run_with_tools(session, registry):
                yield sse_event({"content": chunk}, event="chunk")
            yield sse_event({"content": ""}, event="done")
        except Exception as exc:  # noqa: BLE001
            yield sse_event({"error": str(exc)}, event="error")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
