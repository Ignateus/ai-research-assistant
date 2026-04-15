"""Date and time tool — gives the model access to the current date/time."""

from __future__ import annotations

from datetime import datetime, timezone


def get_current_datetime(timezone_name: str = "UTC") -> str:
    """
    Return the current date and time.

    Args:
        timezone_name: Only "UTC" is supported for now. Future: pytz/zoneinfo zones.

    Returns:
        ISO-8601 datetime string with timezone label.
    """
    now = datetime.now(tz=timezone.utc)
    return now.strftime(f"%Y-%m-%d %H:%M:%S UTC  (weekday: {now.strftime('%A')})")


# --- Tool definition (Anthropic schema) ---

TOOL_DEFINITION = {
    "name": "get_current_datetime",
    "description": (
        "Get the current date and time in UTC. Use this whenever the user asks "
        "about the current date, time, day of the week, or needs a timestamp."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone_name": {
                "type": "string",
                "description": "Timezone to use. Currently only 'UTC' is supported.",
                "default": "UTC",
            }
        },
        "required": [],
    },
}
