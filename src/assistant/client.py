"""Thin wrapper around the Anthropic client with streaming and token tracking."""

from __future__ import annotations

import time
import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, Callable

import anthropic

from .config import config

if TYPE_CHECKING:
    from .tools.registry import ToolRegistry

log = logging.getLogger(__name__)

T = TypeVar("T")

# Retry configuration for rate-limit / transient API errors
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles on each attempt


def _with_retry(fn: Callable[[], T]) -> T:
    """
    Call fn(), retrying up to _MAX_RETRIES times on rate-limit or overload errors.

    Uses exponential backoff: 1s, 2s, 4s between attempts.
    Raises the last exception if all retries are exhausted.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except anthropic.RateLimitError as exc:
            if attempt == _MAX_RETRIES:
                raise
            log.warning("Rate limit hit — retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, _MAX_RETRIES)
            time.sleep(delay)
            delay *= 2
        except anthropic.APIStatusError as exc:
            # Retry on 529 Overloaded; re-raise everything else immediately
            if exc.status_code == 529 and attempt < _MAX_RETRIES:
                log.warning("API overloaded — retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, _MAX_RETRIES)
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("Unreachable")


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
    # Optional extra context (e.g. document summaries) cached separately
    cached_context: str = ""
    history: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def add_user(self, text: str) -> None:
        self.history.append(Message(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.history.append(Message(role="assistant", content=text))

    def to_api_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def build_cached_system(self) -> list[dict]:
        """
        Return a system prompt structured for Anthropic prompt caching.

        Two cache breakpoints are used:
          1. The static system prompt — never changes, always cache-eligible.
          2. The dynamic context block — changes when documents are ingested,
             but stays stable across turns once set.

        Anthropic requires at least 1024 tokens to trigger a cache write.
        Shorter prompts are sent as plain text and still work correctly.
        """
        blocks: list[dict] = [
            {
                "type": "text",
                "text": self.system_prompt or "You are a helpful research assistant.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self.cached_context:
            blocks.append(
                {
                    "type": "text",
                    "text": self.cached_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        return blocks

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
        response = _with_retry(lambda: self._client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system or "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
        ))
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
            system=session.build_cached_system(),
            messages=session.to_api_messages(),
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                yield chunk

            final = stream.get_final_message()
            session.usage.update(final.usage)

        session.add_assistant(full_text)
        return full_text

    # ------------------------------------------------------------------
    # Agentic loop (tool use + prompt caching)
    # ------------------------------------------------------------------

    def run_with_tools(
        self,
        session: ConversationSession,
        registry: "ToolRegistry",
        on_tool_call: callable | None = None,
    ) -> Generator[str, None, None]:
        """
        Agentic loop: keep calling the API until the model stops using tools.

        Yields text chunks as they stream. Tool calls are executed silently
        and their results fed back to the model automatically.
        The system prompt is sent with cache_control on every call so Anthropic
        can serve it from cache after the first turn.

        Args:
            session:      ConversationSession with user message already appended.
            registry:     ToolRegistry with available tools.
            on_tool_call: Optional callback(tool_name, tool_input, result) for
                          displaying tool activity in the UI.
        """
        while True:
            messages = session.to_api_messages()

            response = _with_retry(lambda: self._client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                system=session.build_cached_system(),
                tools=registry.definitions,
                messages=messages,
            ))
            session.usage.update(response.usage)

            full_text = ""
            for block in response.content:
                if block.type == "text":
                    full_text += block.text
                    yield block.text

            if response.stop_reason == "end_turn":
                if full_text:
                    session.add_assistant(full_text)
                break

            if response.stop_reason == "tool_use":
                session.history.append(
                    Message(role="assistant", content=response.content)  # type: ignore[arg-type]
                )

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    result = registry.execute(block.name, block.input)

                    if on_tool_call:
                        on_tool_call(block.name, block.input, result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                session.history.append(
                    Message(role="user", content=tool_results)  # type: ignore[arg-type]
                )
                continue

            break
