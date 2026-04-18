"""Conversation summarizer — compresses old history to stay within context limits."""

from __future__ import annotations

import tiktoken

from ..client import AssistantClient, ConversationSession, Message

# Token threshold before we trigger summarization.
# When history exceeds this, the oldest half of turns are compressed.
DEFAULT_SUMMARIZE_THRESHOLD = 6_000

# How many of the most recent messages to always keep verbatim (never summarized).
# Keeps recent context fresh and avoids summarizing an in-flight conversation.
KEEP_RECENT = 6

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def history_token_count(session: ConversationSession) -> int:
    """Return approximate token count for the full conversation history."""
    return sum(
        count_tokens(m.content if isinstance(m.content, str) else str(m.content))
        for m in session.history
    )


def should_summarize(
    session: ConversationSession,
    threshold: int = DEFAULT_SUMMARIZE_THRESHOLD,
) -> bool:
    """Return True if the conversation history is long enough to warrant summarization."""
    return history_token_count(session) > threshold


def summarize_history(
    session: ConversationSession,
    client: AssistantClient,
    threshold: int = DEFAULT_SUMMARIZE_THRESHOLD,
) -> int:
    """
    If history exceeds the threshold, compress the oldest messages into a summary.

    The most recent KEEP_RECENT messages are always preserved verbatim.
    A synthetic assistant message containing the summary is injected at the start.

    Returns:
        Number of messages that were compressed (0 if no summarization was needed).
    """
    if not should_summarize(session, threshold):
        return 0

    messages = session.history
    if len(messages) <= KEEP_RECENT:
        return 0

    # Split: old messages to compress, recent to keep
    split_point = len(messages) - KEEP_RECENT
    old_messages = messages[:split_point]
    recent_messages = messages[split_point:]

    # Build a transcript of old messages for summarization
    transcript_parts = []
    for msg in old_messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        transcript_parts.append(f"{msg.role.upper()}: {content}")
    transcript = "\n\n".join(transcript_parts)

    summary_prompt = (
        "Below is a conversation transcript. Produce a concise but complete summary "
        "that preserves all key facts, decisions, topics discussed, and any important "
        "context a reader would need to continue the conversation naturally. "
        "Write in past tense. Do not add commentary or opinions.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    summary_text = client.complete(
        prompt=summary_prompt,
        system="You are a precise conversation summarizer.",
    )

    summary_message = Message(
        role="assistant",
        content=(
            f"[Summary of earlier conversation]\n{summary_text}\n"
            "[End of summary — continuing from here]"
        ),
    )

    session.history = [summary_message] + recent_messages
    return len(old_messages)
