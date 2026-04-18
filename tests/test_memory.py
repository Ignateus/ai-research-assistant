"""Unit tests for memory — summarizer and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from assistant.client import ConversationSession, Message, Usage
from assistant.memory.persistence import (
    default_session_path,
    list_sessions,
    load_session,
    save_session,
)
from assistant.memory.summarizer import (
    count_tokens,
    history_token_count,
    should_summarize,
    summarize_history,
)


# ---------------------------------------------------------------------------
# Summarizer helpers
# ---------------------------------------------------------------------------

def test_count_tokens_basic():
    assert count_tokens("hello world") > 0


def test_history_token_count_empty():
    session = ConversationSession()
    assert history_token_count(session) == 0


def test_history_token_count_with_messages():
    session = ConversationSession()
    session.add_user("What is machine learning?")
    session.add_assistant("Machine learning is a subset of AI.")
    assert history_token_count(session) > 0


def test_should_summarize_below_threshold():
    session = ConversationSession()
    session.add_user("Hello")
    assert not should_summarize(session, threshold=1_000_000)


def test_should_summarize_above_threshold():
    session = ConversationSession()
    # Add enough content to exceed a tiny threshold
    for i in range(20):
        session.add_user("word " * 50)
        session.add_assistant("word " * 50)
    assert should_summarize(session, threshold=10)


def test_summarize_history_compresses_old_messages():
    session = ConversationSession()
    # Add 12 messages (exceeds KEEP_RECENT=6)
    for i in range(6):
        session.add_user(f"Question {i}: " + "detail " * 30)
        session.add_assistant(f"Answer {i}: " + "response " * 30)

    mock_client = MagicMock()
    mock_client.complete.return_value = "Summary of earlier conversation."

    compressed = summarize_history(session, mock_client, threshold=1)

    assert compressed > 0
    # History should now start with a summary message
    assert "[Summary of earlier conversation]" in session.history[0].content
    mock_client.complete.assert_called_once()


def test_summarize_history_skips_if_short():
    session = ConversationSession()
    session.add_user("Hi")
    mock_client = MagicMock()
    compressed = summarize_history(session, mock_client, threshold=1_000_000)
    assert compressed == 0
    mock_client.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _make_session() -> ConversationSession:
    session = ConversationSession(
        system_prompt="You are helpful.",
        cached_context="Some cached context.",
    )
    session.add_user("What is AI?")
    session.add_assistant("AI stands for artificial intelligence.")
    session.usage = Usage(input_tokens=100, output_tokens=50)
    return session


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "session.json"
    session = _make_session()
    save_session(session, path)

    loaded = load_session(path)

    assert loaded.system_prompt == session.system_prompt
    assert loaded.cached_context == session.cached_context
    assert len(loaded.history) == len(session.history)
    assert loaded.history[0].role == "user"
    assert loaded.history[0].content == "What is AI?"
    assert loaded.usage.input_tokens == 100
    assert loaded.usage.output_tokens == 50


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "session.json"
    save_session(_make_session(), path)
    assert path.exists()


def test_load_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "nonexistent.json")


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not valid json")
    with pytest.raises(ValueError, match="Invalid session file"):
        load_session(path)


def test_list_sessions_empty(tmp_path):
    assert list_sessions(tmp_path) == []


def test_list_sessions_returns_files(tmp_path):
    for name in ["a.json", "b.json", "c.json"]:
        (tmp_path / name).write_text("{}")
    result = list_sessions(tmp_path)
    assert len(result) == 3


def test_default_session_path_named():
    path = default_session_path("mytest")
    assert "mytest_" in path.name
    assert path.suffix == ".json"


def test_default_session_path_unnamed():
    path = default_session_path()
    assert "session_" in path.name


# ---------------------------------------------------------------------------
# cached_system build
# ---------------------------------------------------------------------------

def test_build_cached_system_no_context():
    session = ConversationSession(system_prompt="Be helpful.")
    blocks = session.build_cached_system()
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Be helpful."
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_build_cached_system_with_context():
    session = ConversationSession(
        system_prompt="Be helpful.",
        cached_context="Loaded: file.txt",
    )
    blocks = session.build_cached_system()
    assert len(blocks) == 2
    assert blocks[1]["text"] == "Loaded: file.txt"
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
