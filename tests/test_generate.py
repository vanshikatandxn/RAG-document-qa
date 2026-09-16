from unittest.mock import MagicMock, patch

from rag_qa.generate import build_context_block, generate_answer
from rag_qa.retrieve import RetrievedChunk


def test_build_context_block_labels_chunks():
    chunks = [
        RetrievedChunk(index=0, text="First chunk text.", score=0.9),
        RetrievedChunk(index=2, text="Second chunk text.", score=0.7),
    ]
    block = build_context_block(chunks)

    assert "[Chunk 0]" in block
    assert "[Chunk 2]" in block
    assert "First chunk text." in block


def test_generate_answer_returns_early_with_no_chunks():
    result = generate_answer("Any question?", [], api_key="fake-key", model="fake-model")
    assert "couldn't find" in result.lower()


@patch("rag_qa.generate.Anthropic")
def test_generate_answer_calls_claude_and_returns_text(mock_anthropic_class):
    # Fake the response Claude's API would return
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The refund window is 30 days. (Source: chunk 1)")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    chunks = [RetrievedChunk(index=1, text="Refunds within 30 days.", score=0.95)]

    result = generate_answer(
        "What is the refund policy?",
        chunks,
        api_key="fake-key",
        model="claude-sonnet-4-5",
    )

    assert "30 days" in result
    mock_client.messages.create.assert_called_once()
