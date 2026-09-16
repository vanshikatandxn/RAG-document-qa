"""
Generation: given a question and the retrieved context chunks, ask
Claude to produce a grounded, cited answer.
"""

from __future__ import annotations

from anthropic import Anthropic

from rag_qa.logging_config import get_logger
from rag_qa.retrieve import RetrievedChunk

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a document Q&A assistant. You will be given \
one or more excerpts from a document, each labeled with a chunk number, \
followed by a question.

Rules:
- Answer using ONLY the information in the provided excerpts.
- If the excerpts do not contain enough information to answer, say so \
explicitly instead of guessing or using outside knowledge.
- At the end of your answer, state which chunk number(s) you used, \
like this: (Source: chunk 2)
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a labeled block for the prompt."""
    return "\n\n".join(f"[Chunk {c.index}]\n{c.text}" for c in chunks)


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    api_key: str,
    model: str,
) -> str:
    """
    Call Claude with the retrieved context and return a grounded,
    cited answer as plain text.
    """
    if not retrieved_chunks:
        return "I couldn't find any relevant content to answer that question."

    context_block = build_context_block(retrieved_chunks)
    user_message = f"{context_block}\n\nQuestion: {question}"

    logger.info("Sending question to %s with %d chunk(s) of context", model, len(retrieved_chunks))

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    logger.info("Received answer (%d characters)", len(answer))
    return answer
