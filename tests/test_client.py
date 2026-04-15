"""Unit tests for the AssistantClient and ConversationSession."""

from unittest.mock import MagicMock, patch

import pytest

from assistant.client import AssistantClient, ConversationSession, Usage


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

def test_usage_totals():
    u = Usage(input_tokens=100, output_tokens=50)
    assert u.total_tokens == 150


def test_usage_str_no_cache():
    u = Usage(input_tokens=10, output_tokens=5)
    assert "in=10" in str(u)
    assert "out=5" in str(u)
    assert "cache" not in str(u)


def test_usage_str_with_cache():
    u = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3)
    assert "cache_read=3" in str(u)


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------

def test_session_history_round_trip():
    s = ConversationSession()
    s.add_user("hello")
    s.add_assistant("hi there")
    msgs = s.to_api_messages()
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_session_clear():
    s = ConversationSession()
    s.add_user("hello")
    s.clear()
    assert s.history == []


# ---------------------------------------------------------------------------
# AssistantClient (mocked)
# ---------------------------------------------------------------------------

@patch("assistant.client.anthropic.Anthropic")
def test_complete_returns_text(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_client.messages.create.return_value = mock_response

    client = AssistantClient()
    result = client.complete("What is AI?")

    assert result == "test response"
    mock_client.messages.create.assert_called_once()
