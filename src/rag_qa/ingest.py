"""
Document ingestion: extract text from a PDF and split it into
retrieval-sized chunks.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from rag_qa.logging_config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF file, page by page.

    Raises:
        FileNotFoundError: if pdf_path doesn't exist.
        ValueError: if the PDF has no extractable text (e.g. a scanned
            image with no OCR layer).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No file found at: {pdf_path}")

    logger.info("Extracting text from %s", pdf_path.name)

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
            logger.info("  page %d: %d characters", i + 1, len(page_text))

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}. "
            f"It may be a scanned/image-only PDF with no text layer."
        )

    return full_text


def chunk_text(text: str, chunk_size_words: int) -> list[str]:
    """
    Split text into chunks of roughly `chunk_size_words` words each.

    This is a simple fixed-size chunker — no overlap, no sentence-boundary
    awareness. That's a deliberate simplification for this scope; a
    production system would typically add ~10-20% overlap between chunks
    so an answer near a chunk boundary doesn't get split awkwardly.
    """
    words = text.split()
    if not words:
        return []

    chunks = [
        " ".join(words[i : i + chunk_size_words])
        for i in range(0, len(words), chunk_size_words)
    ]

    logger.info("Split text into %d chunks (~%d words each)", len(chunks), chunk_size_words)
    return chunks
