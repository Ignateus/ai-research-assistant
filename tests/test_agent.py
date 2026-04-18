"""Unit tests for the agent package."""

from unittest.mock import MagicMock, call, patch

import pytest

from assistant.agent.planner import Plan, Step, create_plan
from assistant.agent.executor import StepResult, execute_step
from assistant.agent.reflector import Reflection, reflect
from assistant.agent.loop import AgentEvent, EventType, run_research_agent


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _mock_client_complete(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def test_create_plan_parses_valid_json():
    json_response = """{
        "steps": [
            {"description": "Search for basics of neural networks", "tool_hint": "web_search"},
            {"description": "Find recent research papers", "tool_hint": "web_search"},
            {"description": "Look up key terminology", "tool_hint": "search_documents"}
        ]
    }"""
    client = _mock_client_complete(json_response)
    plan = create_plan("Explain neural networks", client)

    assert isinstance(plan, Plan)
    assert len(plan.steps) == 3
    assert plan.steps[0].index == 1
    assert plan.steps[0].tool_hint == "web_search"
    assert "neural networks" in plan.steps[0].description.lower()


def test_create_plan_fallback_on_bad_json():
    client = _mock_client_complete("not valid json at all")
    plan = create_plan("Some goal", client)
    # Should not raise — falls back to a single step
    assert len(plan.steps) == 1


def test_create_plan_strips_markdown_fences():
    json_response = '```json\n{"steps": [{"description": "Do a search", "tool_hint": "web_search"}]}\n```'
    client = _mock_client_complete(json_response)
    plan = create_plan("goal", client)
    assert len(plan.steps) == 1


def test_step_str():
    step = Step(index=2, description="Search for papers", tool_hint="web_search")
    assert "Step 2" in str(step)
    assert "web_search" in str(step)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def test_execute_step_returns_findings():
    step = Step(index=1, description="Find info on LLMs", tool_hint="web_search")

    mock_client = MagicMock()
    mock_registry = MagicMock()

    # Simulate run_with_tools yielding text chunks
    mock_client.run_with_tools.return_value = iter(["LLMs are ", "very powerful."])

    result = execute_step(step, mock_client, mock_registry)

    assert isinstance(result, StepResult)
    assert result.findings == "LLMs are very powerful."
    assert result.step == step


def test_execute_step_counts_tool_calls():
    step = Step(index=1, description="Calculate something", tool_hint="calculator")

    mock_client = MagicMock()
    mock_registry = MagicMock()

    # Simulate tool call happening via on_tool_call callback
    def fake_run_with_tools(session, registry, on_tool_call=None):
        if on_tool_call:
            on_tool_call("calculator", {"expression": "2+2"}, "4")
        yield "The answer is 4."

    mock_client.run_with_tools.side_effect = fake_run_with_tools

    result = execute_step(step, mock_client, mock_registry)
    assert result.tool_calls_made == 1


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

def test_reflect_sufficient():
    json_response = '{"sufficient": true, "gaps": [], "additional_steps": []}'
    client = _mock_client_complete(json_response)

    results = [
        StepResult(
            step=Step(index=1, description="Search web"),
            findings="Found lots of relevant info.",
        )
    ]
    reflection = reflect("Research goal", results, client)

    assert reflection.sufficient is True
    assert reflection.gaps == []
    assert reflection.additional_steps == []


def test_reflect_not_sufficient():
    json_response = """{
        "sufficient": false,
        "gaps": ["Missing recent data"],
        "additional_steps": [
            {"description": "Search for recent news", "tool_hint": "web_search"}
        ]
    }"""
    client = _mock_client_complete(json_response)

    results = [StepResult(step=Step(index=1, description="Step"), findings="Partial info.")]
    reflection = reflect("goal", results, client, step_offset=1)

    assert reflection.sufficient is False
    assert len(reflection.gaps) == 1
    assert len(reflection.additional_steps) == 1
    assert reflection.additional_steps[0].index == 2  # step_offset applied


def test_reflect_fallback_on_bad_json():
    client = _mock_client_complete("broken response")
    results = [StepResult(step=Step(index=1, description="Step"), findings="stuff")]
    reflection = reflect("goal", results, client)
    # Should default to sufficient=True rather than crashing
    assert reflection.sufficient is True


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _build_mock_client(plan_json: str, reflect_json: str, report: str) -> MagicMock:
    """Build a mock client with controlled complete() and run_with_tools() responses."""
    client = MagicMock()
    # complete() is called for planning, reflection, and final report
    client.complete.side_effect = [plan_json, reflect_json, report]

    # run_with_tools() is called once per step
    def fake_run_with_tools(session, registry, on_tool_call=None):
        yield "Step findings."

    client.run_with_tools.side_effect = fake_run_with_tools
    return client


def test_run_research_agent_emits_events():
    plan_json = '{"steps": [{"description": "Search web", "tool_hint": "web_search"}]}'
    reflect_json = '{"sufficient": true, "gaps": [], "additional_steps": []}'
    report = "# Report\n\nFindings summary."

    client = _build_mock_client(plan_json, reflect_json, report)
    registry = MagicMock()

    gen = run_research_agent("Test goal", client, registry)
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as e:
        result = e.value

    event_types = [e.type for e in events]
    assert EventType.PLAN_CREATED in event_types
    assert EventType.STEP_STARTED in event_types
    assert EventType.STEP_COMPLETED in event_types
    assert EventType.REFLECTION in event_types
    assert EventType.REPORT_READY in event_types
    assert result.report == report
    assert result.total_steps_executed == 1


def test_run_research_agent_adds_extra_steps_on_gap():
    plan_json = '{"steps": [{"description": "Initial search", "tool_hint": "web_search"}]}'
    reflect_not_sufficient = """{
        "sufficient": false,
        "gaps": ["Missing details"],
        "additional_steps": [{"description": "Extra search", "tool_hint": "web_search"}]
    }"""
    reflect_sufficient = '{"sufficient": true, "gaps": [], "additional_steps": []}'
    report = "# Final Report"

    client = MagicMock()
    client.complete.side_effect = [plan_json, reflect_not_sufficient, reflect_sufficient, report]

    def fake_run_with_tools(session, registry, on_tool_call=None):
        yield "Findings."

    client.run_with_tools.side_effect = fake_run_with_tools
    registry = MagicMock()

    gen = run_research_agent("goal", client, registry)
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as e:
        result = e.value

    extra_events = [e for e in events if e.type == EventType.EXTRA_STEPS_ADDED]
    assert len(extra_events) == 1
    # Should have executed 2 steps total (1 original + 1 extra)
    assert result.total_steps_executed == 2
