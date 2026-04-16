"""Unit tests for the RAG pipeline components."""

import tempfile
from pathlib import Path

import pytest

from assistant.rag.chunker import Chunk, TextChunker
from assistant.rag.loader import Document, load_file, load_directory
from assistant.rag.store import VectorStore
from assistant.rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_load_txt_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello, world!")
    doc = load_file(f)
    assert doc.text == "Hello, world!"
    assert doc.doc_type == "txt"
    assert "test.txt" in doc.source


def test_load_md_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nSome content.")
    doc = load_file(f)
    assert doc.doc_type == "md"
    assert "Title" in doc.text


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_file("/nonexistent/file.txt")


def test_load_unsupported_type(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(f)


def test_load_directory(tmp_path):
    (tmp_path / "a.txt").write_text("file a")
    (tmp_path / "b.md").write_text("file b")
    (tmp_path / "skip.csv").write_text("ignored")
    docs = load_directory(tmp_path)
    assert len(docs) == 2
    assert all(d.doc_type in ("txt", "md") for d in docs)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunker_basic_split():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    # 30 tokens of simple text
    text = " ".join(["word"] * 30)
    chunks = chunker.chunk_text(text, source="test")
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunker_short_text_single_chunk():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "Short text."
    chunks = chunker.chunk_text(text, source="test")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1


def test_chunker_overlap_is_invalid():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_chunker_document():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    doc = Document(text="word " * 200, source="doc.txt", doc_type="txt")
    chunks = chunker.chunk_document(doc)
    assert all(c.source == "doc.txt" for c in chunks)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

def test_vector_store_add_and_query(tmp_path):
    store = VectorStore(collection_name="test_col", persist_dir=str(tmp_path))
    chunks = [
        Chunk(text="The sky is blue.", source="doc.txt", chunk_index=0, total_chunks=2),
        Chunk(text="Photosynthesis converts sunlight into energy.", source="doc.txt", chunk_index=1, total_chunks=2),
    ]
    added = store.add_chunks(chunks)
    assert added == 2
    assert store.count == 2

    results = store.query("What colour is the sky?", top_k=1)
    assert len(results) == 1
    assert "blue" in results[0].text.lower()


def test_vector_store_no_duplicates(tmp_path):
    store = VectorStore(collection_name="test_dedup", persist_dir=str(tmp_path))
    chunk = Chunk(text="Unique content.", source="f.txt", chunk_index=0, total_chunks=1)
    store.add_chunks([chunk])
    added_again = store.add_chunks([chunk])
    assert added_again == 0
    assert store.count == 1


def test_vector_store_empty_query(tmp_path):
    store = VectorStore(collection_name="test_empty", persist_dir=str(tmp_path))
    results = store.query("anything")
    assert results == []


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

def test_pipeline_ingest_and_search(tmp_path):
    f = tmp_path / "knowledge.txt"
    f.write_text(
        "Neural networks are composed of layers of interconnected nodes. "
        "Each node applies a non-linear activation function to its inputs. " * 10
    )

    pipeline = RAGPipeline(persist_dir=str(tmp_path / "chroma"))
    result = pipeline.ingest_file(f)

    assert result.files_loaded == 1
    assert result.chunks_added >= 1

    context = pipeline.search_as_context("What are neural networks made of?")
    assert "node" in context.lower() or "layer" in context.lower()


def test_pipeline_search_as_context_empty(tmp_path):
    pipeline = RAGPipeline(persist_dir=str(tmp_path / "chroma"))
    context = pipeline.search_as_context("anything")
    assert context == ""
