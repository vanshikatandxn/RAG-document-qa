"""
CLI entrypoint: load a PDF, build a retrieval index over it, and let
the user ask questions in a loop, getting grounded, cited answers back.

Usage:
    python -m rag_qa.main path/to/document.pdf
"""

from __future__ import annotations

import argparse
import sys

from rag_qa.config import load_settings
from rag_qa.generate import generate_answer
from rag_qa.ingest import chunk_text, extract_text_from_pdf
from rag_qa.logging_config import get_logger
from rag_qa.retrieve import ChunkRetriever

logger = get_logger(__name__)

EXIT_COMMANDS = {"quit", "exit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF document using RAG."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to load.")
    return parser.parse_args()


def run() -> int:
    args = parse_args()

    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    try:
        text = extract_text_from_pdf(args.pdf_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not load document: {e}", file=sys.stderr)
        return 1

    chunks = chunk_text(text, chunk_size_words=settings.chunk_size_words)
    retriever = ChunkRetriever(chunks)

    print(f"\nLoaded '{args.pdf_path}' — {len(chunks)} chunks ready.")
    print("Ask a question, or type 'quit' to exit.\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            return 0

        retrieved = retriever.retrieve(question, top_k=settings.top_k_chunks)
        answer = generate_answer(
            question,
            retrieved,
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )
        print(f"\n{answer}\n")


if __name__ == "__main__":
    sys.exit(run())
