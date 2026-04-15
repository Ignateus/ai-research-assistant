"""Tool registry — maps tool names to callables and builds Anthropic schemas."""

from __future__ import annotations

from typing import Any, Callable

from . import calculator, datetime_tool, search


class ToolRegistry:
    """
    Central registry of all tools available to the assistant.

    Each tool has:
      - A handler: a callable that takes **kwargs from Claude's tool_use block.
      - A definition: the Anthropic-format schema dict sent with every API request.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., str]] = {}
        self._definitions: list[dict] = []

    def register(self, definition: dict, handler: Callable[..., str]) -> None:
        name = definition["name"]
        self._handlers[name] = handler
        self._definitions.append(definition)

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Run a tool by name and return its string output."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            return handler(**tool_input)
        except Exception as exc:  # noqa: BLE001
            return f"Error running tool '{tool_name}': {exc}"

    @property
    def definitions(self) -> list[dict]:
        """Return the list of tool schemas to pass to the Anthropic API."""
        return self._definitions

    @property
    def names(self) -> list[str]:
        return list(self._handlers.keys())


def build_default_registry() -> ToolRegistry:
    """Build and return the registry with all built-in tools registered."""
    registry = ToolRegistry()
    registry.register(calculator.TOOL_DEFINITION, calculator.calculate)
    registry.register(datetime_tool.TOOL_DEFINITION, datetime_tool.get_current_datetime)
    registry.register(search.TOOL_DEFINITION, search.web_search)
    return registry
