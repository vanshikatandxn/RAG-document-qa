"""
Retrieval: given a question and a list of text chunks, find the
chunks most relevant to the question using TF-IDF + cosine similarity.

Design note: this is a sparse (TF-IDF), not dense (neural embedding),
retrieval method. It's a legitimate, widely-used baseline in real
retrieval systems — fast, free, requires no external API, and easy to
reason about. A production system might layer dense embeddings on top
for semantic matching, but TF-IDF alone is a defensible starting point.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_qa.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    index: int
    text: str
    score: float


class ChunkRetriever:
    """
    Wraps a fitted TF-IDF vectorizer over a fixed set of chunks, so the
    (relatively expensive) fit step happens once, not on every query.
    """

    def __init__(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever with zero chunks.")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)
        logger.info("Built TF-IDF index over %d chunks", len(chunks))

    def retrieve(self, question: str, top_k: int = 2) -> list[RetrievedChunk]:
        """
        Return the top_k chunks most similar to the question, ranked
        highest-score first.
        """
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.chunk_vectors)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = [
            RetrievedChunk(index=int(i), text=self.chunks[i], score=float(scores[i]))
            for i in ranked_indices
        ]

        for r in results:
            logger.info("  retrieved chunk %d (score=%.3f)", r.index, r.score)

        return results
