"""Thin wrapper around the Anthropic client with streaming and token tracking."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, field

import anthropic

from .config import config


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def update(self, usage: anthropic.types.Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        parts = [f"in={self.input_tokens}", f"out={self.output_tokens}"]
        if self.cache_read_tokens:
            parts.append(f"cache_read={self.cache_read_tokens}")
        if self.cache_write_tokens:
            parts.append(f"cache_write={self.cache_write_tokens}")
        return f"Tokens({', '.join(parts)})"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationSession:
    """Holds message history for a multi-turn conversation."""

    system_prompt: str = ""
    history: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def add_user(self, text: str) -> None:
        self.history.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.history.append(Message(role="assistant", content=text))

    def to_api_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def clear(self) -> None:
        self.history.clear()


class AssistantClient:
    """Wraps the Anthropic client with streaming, history, and usage tracking."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=config.api_key)

    # ------------------------------------------------------------------
    # One-shot (no history)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: str = "") -> str:
        """Single-turn, non-streaming completion."""
        response = self._client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system or "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Single-turn streaming completion — yields text chunks."""
        with self._client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system or "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream

    # ------------------------------------------------------------------
    # Multi-turn (with history)
    # ------------------------------------------------------------------

    def chat_stream(self, session: ConversationSession) -> Generator[str, None, str]:
        """
        Stream the next assistant turn for a ConversationSession.

        Yields text chunks. The full assistant reply is appended to
        session.history automatically and returned as the final value.
        """
        full_text = ""

        with self._client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            system=session.system_prompt or "You are a helpful research assistant.",
            messages=session.to_api_messages(),
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                yield chunk

            # Capture usage from the final message
            final = stream.get_final_message()
            session.usage.update(final.usage)

        session.add_assistant(full_text)
        return full_text
