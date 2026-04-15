"""Tools package — function calling support for the research assistant."""

from .registry import ToolRegistry, build_default_registry

__all__ = ["ToolRegistry", "build_default_registry"]
