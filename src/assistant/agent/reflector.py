"""Reflector — evaluates findings against the goal and decides if more steps are needed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..client import AssistantClient
from .executor import StepResult
from .planner import Step

_REFLECTOR_SYSTEM = """\
You are a critical research evaluator. You are given a research goal and the findings \
gathered so far. Your job is to assess whether the findings are sufficient to write a \
comprehensive answer, or whether specific gaps remain.

You MUST respond with a valid JSON object in exactly this format:
{
  "sufficient": true | false,
  "gaps": ["gap description 1", "gap description 2"],
  "additional_steps": [
    {"description": "...", "tool_hint": "web_search|calculator|get_current_datetime|search_documents|none"}
  ]
}

Rules:
- If sufficient is true, gaps and additional_steps should be empty lists.
- If sufficient is false, list 1 to 3 specific gaps and a corresponding additional step for each.
- Be strict: only mark sufficient=true if the goal can be fully addressed with the current findings.
- Respond with JSON only. No prose, no markdown fences.
"""


@dataclass
class Reflection:
    sufficient: bool
    gaps: list[str] = field(default_factory=list)
    additional_steps: list[Step] = field(default_factory=list)

    def __str__(self) -> str:
        if self.sufficient:
            return "Reflection: findings are sufficient to write the report."
        gap_lines = "\n".join(f"  - {g}" for g in self.gaps)
        return f"Reflection: gaps found:\n{gap_lines}"


def reflect(
    goal: str,
    results: list[StepResult],
    client: AssistantClient,
    step_offset: int = 0,
) -> Reflection:
    """
    Evaluate whether collected findings are sufficient to address the research goal.

    Args:
        goal:         The original research goal.
        results:      All StepResults gathered so far.
        client:       AssistantClient instance.
        step_offset:  Used to number any additional steps correctly.

    Returns:
        A Reflection indicating whether more work is needed.
    """
    findings_summary = "\n\n".join(
        f"Step {r.step.index} — {r.step.description}:\n{r.findings}"
        for r in results
    )

    prompt = (
        f"Research goal: {goal}\n\n"
        f"Findings gathered so far:\n\n{findings_summary}"
    )

    raw = client.complete(prompt=prompt, system=_REFLECTOR_SYSTEM)
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(raw)
        additional_steps = [
            Step(
                index=step_offset + i + 1,
                description=s["description"],
                tool_hint=s.get("tool_hint", "none"),
            )
            for i, s in enumerate(data.get("additional_steps", []))
        ]
        return Reflection(
            sufficient=bool(data.get("sufficient", True)),
            gaps=data.get("gaps", []),
            additional_steps=additional_steps,
        )
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, assume we have enough
        return Reflection(sufficient=True)
