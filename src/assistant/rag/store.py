"""ChromaDB vector store — persist, embed, and query document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from ..config import config
from .chunker import Chunk


@dataclass
class SearchResult:
    """A single result returned from a similarity search."""

    text: str
    source: str
    chunk_index: int
    score: float        # lower = more similar (L2 distance)

    def __str__(self) -> str:
        return (
            f"[score={self.score:.3f} | {Path(self.source).name} chunk {self.chunk_index}]\n"
            f"{self.text}"
        )


class VectorStore:
    """
    Thin wrapper around ChromaDB for storing and querying document chunks.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2) so no
    external embedding API or key is required.
    """

    def __init__(self, collection_name: str = "documents", persist_dir: str | None = None) -> None:
        dir_ = persist_dir or config.chroma_dir
        Path(dir_).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=dir_)
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "l2"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to the collection. Skips duplicates by chunk ID.

        Returns the number of chunks actually added.
        """
        if not chunks:
            return 0

        ids = [self._chunk_id(c) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Only insert IDs not already present
        existing = set(self._collection.get(ids=ids)["ids"])
        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]

        if not new_indices:
            return 0

        self._collection.add(
            ids=[ids[i] for i in new_indices],
            documents=[texts[i] for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices],
        )
        return len(new_indices)

    def clear(self) -> None:
        """Remove all documents from the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            embedding_function=self._ef,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Find the most similar chunks to a query string.

        Args:
            text:  The query to search for.
            top_k: Number of results. Defaults to config.top_k_results.

        Returns:
            List of SearchResult ordered by similarity (best first).
        """
        k = top_k or config.top_k_results
        count = self._collection.count()
        if count == 0:
            return []

        k = min(k, count)
        results = self._collection.query(query_texts=[text], n_results=k)

        search_results: list[SearchResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            search_results.append(
                SearchResult(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=dist,
                )
            )
        return search_results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> list[str]:
        """Return a deduplicated list of all ingested source paths."""
        if self.count == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "") for m in all_meta})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        return f"{chunk.source}::{chunk.chunk_index}"
