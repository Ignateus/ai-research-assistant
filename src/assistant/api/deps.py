"""FastAPI dependencies — shared singletons injected via Depends()."""

from __future__ import annotations

from functools import lru_cache

from ..client import AssistantClient
from ..rag import RAGPipeline
from ..tools import build_default_registry
from ..tools.registry import ToolRegistry


@lru_cache(maxsize=1)
def get_client() -> AssistantClient:
    """Return a single shared AssistantClient for the lifetime of the process."""
    return AssistantClient()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Return a single shared RAGPipeline (persists ChromaDB state across requests)."""
    return RAGPipeline()


def get_registry() -> ToolRegistry:
    """Build a fresh ToolRegistry for each request, wired to the shared pipeline."""
    pipeline = get_pipeline()
    return build_default_registry(pipeline=pipeline)
