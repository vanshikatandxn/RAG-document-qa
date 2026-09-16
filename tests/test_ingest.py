from rag_qa.ingest import chunk_text


def test_chunk_text_splits_by_word_count():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size_words=4)

    assert len(chunks) == 3
    assert chunks[0] == "word0 word1 word2 word3"
    assert chunks[-1] == "word8 word9"


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size_words=100) == []
