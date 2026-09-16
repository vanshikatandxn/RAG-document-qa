from rag_qa.retrieve import ChunkRetriever


def test_retrieve_finds_most_relevant_chunk():
    chunks = [
        "The company was founded in 2015 and is based in Chennai.",
        "Our refund policy allows returns within 30 days of purchase.",
        "The product comes in three colors: red, blue, and black.",
    ]
    retriever = ChunkRetriever(chunks)

    results = retriever.retrieve("What is the refund policy?", top_k=1)

    assert len(results) == 1
    assert results[0].index == 1
    assert "refund" in results[0].text.lower()


def test_retrieve_respects_top_k():
    chunks = ["chunk one about cats", "chunk two about dogs", "chunk three about birds"]
    retriever = ChunkRetriever(chunks)

    results = retriever.retrieve("tell me about pets", top_k=2)

    assert len(results) == 2
