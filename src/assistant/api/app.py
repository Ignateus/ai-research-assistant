"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .models import HealthResponse
from .routes import chat, documents, research


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up shared singletons on startup."""
    from .deps import get_client, get_pipeline
    get_client()
    get_pipeline()
    yield
    # Teardown (nothing needed — ChromaDB flushes automatically)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Research Assistant API",
        description=(
            "A research assistant powered by Claude. "
            "Supports streaming chat, multi-step agent research, and document Q&A via RAG."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # CORS — allow all origins for local dev; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(chat.router)
    app.include_router(research.router)
    app.include_router(documents.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        """Check that the API is running."""
        return HealthResponse(status="ok", version=__version__)

    return app


# Module-level app instance used by uvicorn
app = create_app()
