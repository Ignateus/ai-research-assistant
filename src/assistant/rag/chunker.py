"""Text chunker — splits documents into overlapping chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from .loader import Document


@dataclass
class Chunk:
    """A slice of a document ready for embedding."""

    text: str
    source: str      # inherited from the parent Document
    chunk_index: int
    total_chunks: int

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


class TextChunker:
    """
    Splits text into overlapping token-based chunks.

    Uses tiktoken for accurate token counting so chunks stay within
    embedding model limits regardless of language or whitespace density.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(encoding)

    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        """Split a raw string into Chunk objects."""
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        windows = [
            tokens[i : i + self.chunk_size]
            for i in range(0, len(tokens), step)
        ]
        # Remove empty trailing window that can appear on exact boundaries
        windows = [w for w in windows if w]

        total = len(windows)
        return [
            Chunk(
                text=self._enc.decode(window),
                source=source,
                chunk_index=idx,
                total_chunks=total,
            )
            for idx, window in enumerate(windows)
        ]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a loaded Document."""
        return self.chunk_text(doc.text, source=doc.source)

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a list of Documents, flattening into a single list."""
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunk_document(doc))
        return chunks
