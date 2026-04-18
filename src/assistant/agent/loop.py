"""Agent loop — orchestrates plan → execute → reflect → report."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum, auto

from ..client import AssistantClient
from ..tools.registry import ToolRegistry
from .executor import StepResult, execute_step
from .planner import Plan, Step, create_plan
from .reflector import Reflection, reflect

# Maximum reflection rounds to prevent infinite loops
MAX_REFLECTION_ROUNDS = 2

_REPORT_SYSTEM = """\
You are an expert research writer. Given a research goal and a set of findings, \
produce a comprehensive, well-structured markdown report.

The report must include:
- A clear title (# heading)
- An executive summary (2–3 sentences)
- Clearly labelled sections covering all key findings (## headings)
- A "Limitations & Caveats" section noting knowledge gaps or uncertainty
- A "Sources Used" section listing the tools and queries used

Write in clear, professional prose. Use bullet points only where they aid readability.
"""


# ---------------------------------------------------------------------------
# Events — emitted by the loop for the UI to consume
# ---------------------------------------------------------------------------

class EventType(Enum):
    PLAN_CREATED = auto()
    STEP_STARTED = auto()
    STEP_COMPLETED = auto()
    REFLECTION = auto()
    EXTRA_STEPS_ADDED = auto()
    REPORT_READY = auto()


@dataclass
class AgentEvent:
    type: EventType
    data: object = None


@dataclass
class AgentResult:
    goal: str
    plan: Plan
    results: list[StepResult]
    reflections: list[Reflection]
    report: str
    total_steps_executed: int

    def __str__(self) -> str:
        return self.report


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_research_agent(
    goal: str,
    client: AssistantClient,
    registry: ToolRegistry,
) -> Generator[AgentEvent, None, AgentResult]:
    """
    Run the full research agent loop for a given goal.

    Yields AgentEvents so callers can display live progress.
    Returns the final AgentResult.

    Phases:
      1. Plan   — decompose goal into steps
      2. Execute — run each step with tools
      3. Reflect — check for gaps (up to MAX_REFLECTION_ROUNDS)
      4. Report  — synthesise findings into a markdown report
    """
    all_results: list[StepResult] = []
    all_reflections: list[Reflection] = []

    # ── Phase 1: Plan ────────────────────────────────────────────────────
    plan = create_plan(goal, client)
    yield AgentEvent(type=EventType.PLAN_CREATED, data=plan)

    steps_to_run = list(plan.steps)

    # ── Phase 2 + 3: Execute → Reflect (with optional extra rounds) ──────
    reflection_round = 0

    while steps_to_run:
        for step in steps_to_run:
            yield AgentEvent(type=EventType.STEP_STARTED, data=step)
            result = execute_step(step, client, registry)
            all_results.append(result)
            yield AgentEvent(type=EventType.STEP_COMPLETED, data=result)

        steps_to_run = []  # Clear — will be repopulated if reflection finds gaps

        if reflection_round < MAX_REFLECTION_ROUNDS:
            reflection_round += 1
            reflection = reflect(
                goal=goal,
                results=all_results,
                client=client,
                step_offset=len(all_results),
            )
            all_reflections.append(reflection)
            yield AgentEvent(type=EventType.REFLECTION, data=reflection)

            if not reflection.sufficient and reflection.additional_steps:
                steps_to_run = reflection.additional_steps
                yield AgentEvent(type=EventType.EXTRA_STEPS_ADDED, data=steps_to_run)

    # ── Phase 4: Report ──────────────────────────────────────────────────
    findings_text = "\n\n---\n\n".join(
        f"**Step {r.step.index}: {r.step.description}**\n\n{r.findings}"
        for r in all_results
    )
    report_prompt = (
        f"Research goal: {goal}\n\n"
        f"Findings:\n\n{findings_text}"
    )
    report = client.complete(prompt=report_prompt, system=_REPORT_SYSTEM)

    yield AgentEvent(type=EventType.REPORT_READY, data=report)

    return AgentResult(
        goal=goal,
        plan=plan,
        results=all_results,
        reflections=all_reflections,
        report=report,
        total_steps_executed=len(all_results),
    )
