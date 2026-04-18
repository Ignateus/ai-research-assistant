"""Documents routes — ingest files, list sources, clear the store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...rag import RAGPipeline
from ..deps import get_pipeline
from ..models import ClearResponse, IngestResponse, SourcesResponse

router = APIRouter(prefix="/documents", tags=["documents"])

_SUPPORTED_TYPES = {".txt", ".md", ".pdf"}


@router.post(
    "/ingest",
    summary="Upload and ingest a document",
    response_model=IngestResponse,
)
async def ingest_document(
    file: UploadFile = File(..., description="A .txt, .md, or .pdf file to ingest"),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """
    Upload a document and add it to the vector store.

    Supported formats: `.txt`, `.md`, `.pdf`

    Duplicate chunks (same file ingested twice) are silently skipped.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type {suffix!r}. Supported: {sorted(_SUPPORTED_TYPES)}",
        )

    # Write upload to a temp file so the loader can read it by path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = pipeline.ingest_file(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        files_loaded=result.files_loaded,
        chunks_added=result.chunks_added,
        chunks_skipped=result.chunks_skipped,
        message=str(result),
    )


@router.get(
    "/sources",
    summary="List ingested document sources",
    response_model=SourcesResponse,
)
async def list_sources(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> SourcesResponse:
    """Return a list of all document sources currently in the vector store."""
    sources = pipeline.list_sources()
    return SourcesResponse(count=len(sources), sources=sources)


@router.delete(
    "",
    summary="Clear all ingested documents",
    response_model=ClearResponse,
)
async def clear_documents(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ClearResponse:
    """Remove all documents from the vector store."""
    pipeline.clear()
    return ClearResponse(message="Document store cleared.")
