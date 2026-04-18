"""Uvicorn entry point — `serve` CLI command."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "assistant.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
