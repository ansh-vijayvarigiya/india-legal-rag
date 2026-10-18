from legalrag.chunk import chunk_document
from legalrag.models import Document, make_chunk_id


def _doc(text: str) -> Document:
    return Document("rbi:1", "rbi", "https://example.invalid/1", "T", None, text, {})


def test_chunks_are_contiguous_and_within_budget():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(6))
    chunks = chunk_document(_doc(text), max_chars=1000, overlap_chars=100)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [make_chunk_id("rbi:1", i) for i in range(len(chunks))]
    assert all(len(c.text) <= 1000 for c in chunks)


def test_overlap_carries_tail_text_into_the_next_chunk():
    text = "\n\n".join("word " * 100 for _ in range(6))
    chunks = chunk_document(_doc(text), max_chars=600, overlap_chars=200)
    assert len(chunks) > 1
    assert chunks[0].text[-100:] in chunks[1].text


def test_embed_text_equals_text_at_this_stage():
    chunks = chunk_document(_doc("A single short paragraph."))
    assert chunks[0].embed_text == chunks[0].text


def test_blank_document_produces_no_chunks():
    assert chunk_document(_doc("")) == []
    assert chunk_document(_doc("   \n\n  ")) == []
