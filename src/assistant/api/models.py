"""Pydantic models for API request and response bodies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    system: str | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "messages": [{"role": "user", "content": "What is retrieval-augmented generation?"}]
        }
    }}


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    goal: str = Field(..., min_length=5)

    model_config = {"json_schema_extra": {
        "example": {"goal": "Explain the current state of LLM-based AI agents"}
    }}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    files_loaded: int
    chunks_added: int
    chunks_skipped: int
    message: str


class SourcesResponse(BaseModel):
    count: int
    sources: list[str]


class ClearResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
