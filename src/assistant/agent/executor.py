"""Executor — runs a single plan step using the tool registry."""

from __future__ import annotations

from dataclasses import dataclass

from ..client import AssistantClient, ConversationSession
from ..tools.registry import ToolRegistry
from .planner import Step

_EXECUTOR_SYSTEM = """\
You are a focused research executor. You are given one specific research step to complete.
Use the available tools to gather the information requested in the step.
Be thorough but concise — return only the findings relevant to the step.
Do not editorialize or add commentary beyond the facts you find.
"""


@dataclass
class StepResult:
    step: Step
    findings: str
    tool_calls_made: int = 0

    def __str__(self) -> str:
        return f"[Step {self.step.index}] {self.step.description}\n{self.findings}"


def execute_step(
    step: Step,
    client: AssistantClient,
    registry: ToolRegistry,
) -> StepResult:
    """
    Execute a single plan step by running an agentic tool loop for that step.

    The executor is given a fresh single-turn session scoped to just the step,
    so tool calls and findings are isolated per step.

    Args:
        step:     The Step to execute.
        client:   AssistantClient instance.
        registry: ToolRegistry with available tools.

    Returns:
        A StepResult with the findings text and tool call count.
    """
    session = ConversationSession(system_prompt=_EXECUTOR_SYSTEM)

    # Give the executor a focused prompt for this step
    prompt = f"Complete this research step:\n\n{step.description}"
    if step.tool_hint and step.tool_hint != "none":
        prompt += f"\n\nPreferred tool: {step.tool_hint}"

    session.add_user(prompt)

    findings_parts: list[str] = []
    tool_count = 0

    def _count_tool(name: str, inputs: dict, result: str) -> None:
        nonlocal tool_count
        tool_count += 1

    for chunk in client.run_with_tools(session, registry, on_tool_call=_count_tool):
        findings_parts.append(chunk)

    findings = "".join(findings_parts).strip()
    return StepResult(step=step, findings=findings, tool_calls_made=tool_count)
