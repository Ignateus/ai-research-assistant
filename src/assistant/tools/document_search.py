"""Document search tool — queries the RAG pipeline for relevant context."""

from __future__ import annotations

# The pipeline instance is injected at runtime via set_pipeline()
# so the tool function itself stays stateless and easily testable.
_pipeline = None


def set_pipeline(pipeline) -> None:  # noqa: ANN001
    """Inject a RAGPipeline instance. Call this before registering the tool."""
    global _pipeline
    _pipeline = pipeline


def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search ingested documents for content relevant to a query.

    Args:
        query: The question or topic to search for.
        top_k: Number of chunks to retrieve (default 5).

    Returns:
        Formatted context string, or a message if no documents are loaded.
    """
    if _pipeline is None:
        return "No documents have been ingested yet. Use /ingest <path> to load documents."

    if _pipeline.document_count == 0:
        return "The document store is empty. Use /ingest <path> to load documents first."

    context = _pipeline.search_as_context(query, top_k=top_k)
    if not context:
        return f"No relevant content found for query: {query!r}"

    return context


# --- Tool definition (Anthropic schema) ---

TOOL_DEFINITION = {
    "name": "search_documents",
    "description": (
        "Search through ingested documents to find relevant information. "
        "Use this when the user asks questions about documents they have loaded, "
        "or when you need specific factual context from uploaded files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or topic to search for in the documents.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of relevant chunks to retrieve (1–10). Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
