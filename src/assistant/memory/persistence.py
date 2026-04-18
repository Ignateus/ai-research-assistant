"""Session persistence — save and load conversation sessions to/from JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..client import ConversationSession, Message


def save_session(session: ConversationSession, path: str | Path) -> None:
    """
    Serialise a ConversationSession to a JSON file.

    Args:
        session: The session to save.
        path:    File path to write (created if it doesn't exist).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _serialise_content(content) -> str:  # noqa: ANN001
        """Convert content to a JSON-safe string."""
        if isinstance(content, str):
            return content
        # Tool-use blocks are lists of dicts — serialise to JSON string
        return json.dumps(content)

    data = {
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "system_prompt": session.system_prompt,
        "cached_context": session.cached_context,
        "history": [
            {"role": m.role, "content": _serialise_content(m.content)}
            for m in session.history
        ],
        "usage": {
            "input_tokens": session.usage.input_tokens,
            "output_tokens": session.usage.output_tokens,
            "cache_read_tokens": session.usage.cache_read_tokens,
            "cache_write_tokens": session.usage.cache_write_tokens,
        },
    }

    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_session(path: str | Path) -> ConversationSession:
    """
    Deserialise a ConversationSession from a JSON file.

    Args:
        path: File path to read.

    Returns:
        A restored ConversationSession.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid session file: {exc}") from exc

    from ..client import Usage

    session = ConversationSession(
        system_prompt=data.get("system_prompt", ""),
        cached_context=data.get("cached_context", ""),
        history=[
            Message(role=m["role"], content=m["content"])
            for m in data.get("history", [])
        ],
        usage=Usage(
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            cache_read_tokens=data.get("usage", {}).get("cache_read_tokens", 0),
            cache_write_tokens=data.get("usage", {}).get("cache_write_tokens", 0),
        ),
    )
    return session


def list_sessions(directory: str | Path = "data/sessions") -> list[Path]:
    """Return all saved session files in a directory, newest first."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def default_session_path(name: str | None = None) -> Path:
    """Return a timestamped default path for a new session file."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{ts}.json" if name else f"session_{ts}.json"
    return Path("data/sessions") / filename
