"""Planner — decomposes a research goal into a concrete list of steps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..client import AssistantClient


@dataclass
class Step:
    index: int
    description: str
    tool_hint: str = ""   # Which tool the executor should prefer for this step

    def __str__(self) -> str:
        hint = f" [{self.tool_hint}]" if self.tool_hint else ""
        return f"Step {self.index}{hint}: {self.description}"


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Goal: {self.goal}", ""]
        for step in self.steps:
            lines.append(f"  {step}")
        return "\n".join(lines)


_PLANNER_SYSTEM = """\
You are a research planning assistant. When given a research goal, you produce a \
concise, actionable step-by-step plan to gather the information needed to address it.

You MUST respond with a valid JSON object in exactly this format:
{
  "steps": [
    {"description": "...", "tool_hint": "web_search|calculator|get_current_datetime|search_documents|none"},
    ...
  ]
}

Rules:
- Produce 3 to 6 steps. No more, no less.
- Each step must be specific and actionable — something an agent can execute with one tool call.
- tool_hint should name the single most relevant tool, or "none" if no tool is needed.
- Do not include a final "write report" step — that is handled separately.
- Respond with JSON only. No prose, no markdown fences.
"""


def create_plan(goal: str, client: AssistantClient) -> Plan:
    """
    Ask Claude to decompose a research goal into a list of executable steps.

    Args:
        goal:   The high-level research goal.
        client: AssistantClient to use for the planning call.

    Returns:
        A Plan with 3–6 Steps.
    """
    prompt = f"Research goal: {goal}"
    raw = client.complete(prompt=prompt, system=_PLANNER_SYSTEM)

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(raw)
        steps = [
            Step(
                index=i + 1,
                description=s["description"],
                tool_hint=s.get("tool_hint", "none"),
            )
            for i, s in enumerate(data["steps"])
        ]
    except (json.JSONDecodeError, KeyError) as exc:
        # Fallback: treat the whole response as a single step
        steps = [Step(index=1, description=raw[:300], tool_hint="none")]

    return Plan(goal=goal, steps=steps)
