"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import json

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return a JSON string of results.

    Each result contains: title, url, snippet.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string with a list of results, or an error message.
    """
    if not _DDGS_AVAILABLE:
        return (
            "Search unavailable: install 'duckduckgo-search' package. "
            "Run: pip install duckduckgo-search"
        )

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        return json.dumps(results, indent=2)
    except Exception as exc:  # noqa: BLE001
        return f"Search error: {exc}"


# --- Tool definition (Anthropic schema) ---

TOOL_DEFINITION = {
    "name": "web_search",
    "description": (
        "Search the web for current information on a topic. "
        "Returns a list of results with title, URL, and a short snippet. "
        "Use this to find up-to-date facts, news, or references."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1–10). Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
