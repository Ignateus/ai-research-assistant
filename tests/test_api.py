"""Tests for the FastAPI routes."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.api.deps import get_client, get_pipeline, get_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client():
    client = MagicMock()
    # Default: run_with_tools yields a single text chunk
    client.run_with_tools.return_value = iter(["Hello from the assistant."])
    return client


@pytest.fixture()
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.document_count = 0
    pipeline.list_sources.return_value = []
    return pipeline


@pytest.fixture()
def mock_registry():
    return MagicMock()


@pytest.fixture()
def api_client(mock_client, mock_pipeline, mock_registry):
    """TestClient with all shared dependencies overridden."""
    app = create_app()
    app.dependency_overrides[get_client] = lambda: mock_client
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_registry] = lambda: mock_registry
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def test_chat_returns_sse_stream(api_client, mock_client):
    mock_client.run_with_tools.return_value = iter(["Hello ", "world!"])

    response = api_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    raw = response.text
    assert "chunk" in raw
    assert "Hello" in raw
    assert "done" in raw


def test_chat_requires_messages(api_client):
    response = api_client.post("/chat", json={"messages": []})
    assert response.status_code == 422


def test_chat_accepts_custom_system(api_client, mock_client):
    mock_client.run_with_tools.return_value = iter(["Response."])
    response = api_client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a pirate.",
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

def _make_research_client(plan_json, reflect_json, report):
    client = MagicMock()
    client.complete.side_effect = [plan_json, reflect_json, report]

    def fake_run_with_tools(session, registry, on_tool_call=None):
        yield "Some findings."

    client.run_with_tools.side_effect = fake_run_with_tools
    return client


def test_research_streams_events(mock_pipeline, mock_registry):
    plan_json = '{"steps": [{"description": "Search for info", "tool_hint": "web_search"}]}'
    reflect_json = '{"sufficient": true, "gaps": [], "additional_steps": []}'
    report = "# Research Report\n\nFindings."

    client = _make_research_client(plan_json, reflect_json, report)

    app = create_app()
    app.dependency_overrides[get_client] = lambda: client
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_registry] = lambda: mock_registry

    with TestClient(app) as tc:
        response = tc.post("/research", json={"goal": "What are LLM agents?"})

    assert response.status_code == 200
    raw = response.text
    assert "plan" in raw
    assert "step_started" in raw
    assert "report" in raw
    assert "done" in raw


def test_research_goal_too_short(api_client):
    response = api_client.post("/research", json={"goal": "AI"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def test_list_sources_empty(api_client, mock_pipeline):
    mock_pipeline.list_sources.return_value = []
    response = api_client.get("/documents/sources")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "sources": []}


def test_list_sources_with_data(api_client, mock_pipeline):
    mock_pipeline.list_sources.return_value = ["file_a.txt", "file_b.md"]
    response = api_client.get("/documents/sources")
    data = response.json()
    assert data["count"] == 2
    assert "file_a.txt" in data["sources"]


def test_clear_documents(api_client, mock_pipeline):
    response = api_client.delete("/documents")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"].lower()
    mock_pipeline.clear.assert_called_once()


def test_ingest_unsupported_type(api_client):
    response = api_client.post(
        "/documents/ingest",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 422
    assert "Unsupported" in response.json()["detail"]


def test_ingest_txt_file(api_client, mock_pipeline):
    mock_result = MagicMock()
    mock_result.files_loaded = 1
    mock_result.chunks_added = 3
    mock_result.chunks_skipped = 0
    mock_result.__str__ = lambda self: "Ingested 1 file(s) — 3 new chunks added, 0 already existed."
    mock_pipeline.ingest_file.return_value = mock_result

    response = api_client.post(
        "/documents/ingest",
        files={"file": ("notes.txt", b"This is a test document with some content.", "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["files_loaded"] == 1
    assert data["chunks_added"] == 3
