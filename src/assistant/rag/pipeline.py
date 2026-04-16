"""RAG pipeline — combines loader, chunker, and vector store into one interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import config
from .chunker import TextChunker
from .loader import Document, load_directory, load_file
from .store import SearchResult, VectorStore


@dataclass
class IngestResult:
    files_loaded: int
    chunks_added: int
    chunks_skipped: int

    def __str__(self) -> str:
        return (
            f"Ingested {self.files_loaded} file(s) — "
            f"{self.chunks_added} new chunks added, {self.chunks_skipped} already existed."
        )


class RAGPipeline:
    """
    High-level interface for the full RAG workflow:

        ingest(path)  →  loads, chunks, embeds, stores
        search(query) →  retrieves relevant chunks as formatted context
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        collection_name: str = "documents",
        persist_dir: str | None = None,
    ) -> None:
        self._chunker = TextChunker(
            chunk_size=chunk_size or config.chunk_size,
            chunk_overlap=chunk_overlap or config.chunk_overlap,
        )
        self._store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> IngestResult:
        """Load, chunk, and store a single file."""
        doc = load_file(path)
        return self._ingest_documents([doc])

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> IngestResult:
        """Load, chunk, and store all supported files in a directory."""
        docs = load_directory(directory, recursive=recursive)
        return self._ingest_documents(docs)

    def _ingest_documents(self, docs: list[Document]) -> IngestResult:
        chunks = self._chunker.chunk_documents(docs)
        total = len(chunks)
        added = self._store.add_chunks(chunks)
        return IngestResult(
            files_loaded=len(docs),
            chunks_added=added,
            chunks_skipped=total - added,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the most relevant chunks for a query."""
        return self._store.query(query, top_k=top_k)

    def search_as_context(self, query: str, top_k: int | None = None) -> str:
        """
        Return retrieved chunks formatted as a context block for the LLM.

        Returns an empty string if no documents have been ingested.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        parts = ["Relevant context from ingested documents:\n"]
        for i, r in enumerate(results, 1):
            source_name = Path(r.source).name
            parts.append(f"--- Source {i}: {source_name} (chunk {r.chunk_index}) ---")
            parts.append(r.text)
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return self._store.count

    def list_sources(self) -> list[str]:
        return self._store.sources()

    def clear(self) -> None:
        """Wipe all stored documents."""
        self._store.clear()
