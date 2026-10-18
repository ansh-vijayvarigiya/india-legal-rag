from legalrag.golden_candidates import render_candidates, sample_chunks
from legalrag.models import Chunk


def test_sample_chunks_is_deterministic_and_bounded():
    chunks = [Chunk(f"d#{i:04d}", "d", i, f"text {i}", f"text {i}", {}) for i in range(50)]
    first = sample_chunks(chunks, 10, seed=7)
    assert first == sample_chunks(chunks, 10, seed=7)
    assert len(first) == 10
    assert all(c in chunks for c in first)


def test_render_candidates_includes_chunk_ids_and_text():
    chunks = [Chunk("d#0000", "d", 0, "alpha text", "alpha text", {})]
    rendered = render_candidates(chunks)
    assert "d#0000" in rendered
    assert "alpha text" in rendered


def test_render_candidates_shows_citation_when_available():
    chunks = [Chunk("d#0000", "d", 0, "t", "t", {"citation": "FEMA, Rule 3"})]
    assert "FEMA, Rule 3" in render_candidates(chunks)
