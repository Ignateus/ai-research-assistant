"""Document loader — reads .txt, .md, and .pdf files into raw text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """A loaded document with its text content and source metadata."""

    text: str
    source: str          # file path or identifier
    doc_type: str        # "txt", "md", "pdf"

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, chars={len(self.text)}, preview={preview!r})"


def load_file(path: str | Path) -> Document:
    """
    Load a single file and return a Document.

    Supported formats: .txt, .md, .pdf
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        return Document(text=text, source=str(p), doc_type=suffix.lstrip("."))

    if suffix == ".pdf":
        return _load_pdf(p)

    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .txt, .md, .pdf")


def load_directory(directory: str | Path, recursive: bool = True) -> list[Document]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to scan.
        recursive: If True, also scan subdirectories.

    Returns:
        List of Document objects, sorted by file path.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    supported = {".txt", ".md", ".pdf"}

    docs: list[Document] = []
    for p in sorted(d.glob(pattern)):
        if p.is_file() and p.suffix.lower() in supported:
            try:
                docs.append(load_file(p))
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: skipping {p} — {exc}")

    return docs


def _load_pdf(path: Path) -> Document:
    try:
        import pypdf  # optional dependency

        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        return Document(text=text, source=str(path), doc_type="pdf")
    except ImportError:
        raise ImportError(
            "PDF support requires the 'pypdf' package. Install it with: pip install pypdf"
        )
