"""Centralised configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Anthropic
    api_key: str = os.environ["ANTHROPIC_API_KEY"]
    model: str = os.getenv("MODEL", "claude-sonnet-4-6")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "8096"))

    # RAG (used from Day 3 onwards)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Paths
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")


config = Config()
