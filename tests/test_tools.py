"""Unit tests for the tools package."""

from unittest.mock import MagicMock, patch

import pytest

from assistant.tools.calculator import calculate
from assistant.tools.datetime_tool import get_current_datetime
from assistant.tools.registry import ToolRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,expected", [
    ("2 + 2", "4"),
    ("10 - 3", "7"),
    ("3 * 4", "12"),
    ("10 / 4", "2.5"),
    ("2 ** 10", "1024"),
    ("10 % 3", "1"),
    ("sqrt(144)", "12"),
    ("round(3.14159, 2)", "3.14"),
])
def test_calculator_basic(expression, expected):
    assert calculate(expression) == expected


def test_calculator_division_by_zero():
    result = calculate("1 / 0")
    assert "division by zero" in result.lower()


def test_calculator_invalid_expression():
    result = calculate("import os")
    assert "error" in result.lower()


def test_calculator_pi_constant():
    result = calculate("round(pi, 5)")
    assert result == "3.14159"


# ---------------------------------------------------------------------------
# Datetime
# ---------------------------------------------------------------------------

def test_get_current_datetime_returns_utc():
    result = get_current_datetime()
    assert "UTC" in result


def test_get_current_datetime_has_weekday():
    result = get_current_datetime()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert any(day in result for day in days)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

def test_registry_execute_known_tool():
    registry = ToolRegistry()
    registry.register(
        {"name": "echo", "description": "echo", "input_schema": {"type": "object", "properties": {}, "required": []}},
        lambda text="": text,
    )
    assert registry.execute("echo", {"text": "hello"}) == "hello"


def test_registry_execute_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("nonexistent", {})
    assert "unknown tool" in result.lower()


def test_build_default_registry_has_all_tools():
    registry = build_default_registry()
    assert "calculator" in registry.names
    assert "get_current_datetime" in registry.names
    assert "web_search" in registry.names


def test_registry_definitions_are_valid_schemas():
    registry = build_default_registry()
    for defn in registry.definitions:
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn
