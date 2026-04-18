"""Research route — POST /research runs the agent loop and streams events as SSE."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...agent import EventType, run_research_agent
from ...client import AssistantClient
from ...tools.registry import ToolRegistry
from ..deps import get_client, get_registry
from ..models import ResearchRequest
from ..sse import sse_event

router = APIRouter(prefix="/research", tags=["research"])


@router.post(
    "",
    summary="Run the research agent (streaming)",
    response_description="Server-Sent Events stream of agent progress and final report",
)
async def research(
    request: ResearchRequest,
    client: AssistantClient = Depends(get_client),
    registry: ToolRegistry = Depends(get_registry),
) -> StreamingResponse:
    """
    Run the multi-step research agent on a goal and stream progress events.

    The response is a stream of Server-Sent Events:

    | Event          | Payload fields                              |
    |----------------|---------------------------------------------|
    | `plan`         | `goal`, `steps` (list of step descriptions) |
    | `step_started` | `index`, `description`, `tool_hint`         |
    | `step_done`    | `index`, `findings_preview`                 |
    | `reflection`   | `sufficient`, `gaps`                        |
    | `extra_steps`  | `count`                                     |
    | `report`       | `markdown` (full report text)               |
    | `done`         | `total_steps`                               |
    | `error`        | `error` (message string)                    |
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            gen = run_research_agent(request.goal, client, registry)
            result = None

            try:
                while True:
                    event = next(gen)

                    if event.type == EventType.PLAN_CREATED:
                        plan = event.data
                        yield sse_event(
                            {
                                "goal": plan.goal,
                                "steps": [
                                    {"index": s.index, "description": s.description, "tool_hint": s.tool_hint}
                                    for s in plan.steps
                                ],
                            },
                            event="plan",
                        )

                    elif event.type == EventType.STEP_STARTED:
                        step = event.data
                        yield sse_event(
                            {"index": step.index, "description": step.description, "tool_hint": step.tool_hint},
                            event="step_started",
                        )

                    elif event.type == EventType.STEP_COMPLETED:
                        r = event.data
                        preview = r.findings[:200].replace("\n", " ")
                        yield sse_event(
                            {"index": r.step.index, "findings_preview": preview},
                            event="step_done",
                        )

                    elif event.type == EventType.REFLECTION:
                        reflection = event.data
                        yield sse_event(
                            {"sufficient": reflection.sufficient, "gaps": reflection.gaps},
                            event="reflection",
                        )

                    elif event.type == EventType.EXTRA_STEPS_ADDED:
                        yield sse_event({"count": len(event.data)}, event="extra_steps")

                    elif event.type == EventType.REPORT_READY:
                        yield sse_event({"markdown": event.data}, event="report")

            except StopIteration as e:
                result = e.value

            total = result.total_steps_executed if result else 0
            yield sse_event({"total_steps": total}, event="done")

        except Exception as exc:  # noqa: BLE001
            yield sse_event({"error": str(exc)}, event="error")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
