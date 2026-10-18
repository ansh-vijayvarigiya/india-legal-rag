from legalrag.corpus import read_chunks, write_chunks
from legalrag.models import Chunk


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunks = [
        Chunk("rbi:1#0000", "rbi:1", 0, "alpha", "alpha", {"a": "b"}),
        Chunk("rbi:1#0001", "rbi:1", 1, "beta", "beta", {}),
    ]
    assert write_chunks(path, chunks) == 2
    assert list(read_chunks(path)) == chunks


def test_read_chunks_missing_file_returns_empty(tmp_path):
    assert list(read_chunks(tmp_path / "absent.jsonl")) == []
