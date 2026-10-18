import numpy as np

from legalrag.index import ChunkIndex
from legalrag.models import Chunk

DIM = 4


def _chunk(i: int) -> Chunk:
    return Chunk(f"d#{i:04d}", "d", i, f"text {i}", f"text {i}", {})


def test_upsert_and_search_returns_ranked_chunks(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    chunks = [_chunk(0), _chunk(1)]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    index.upsert(chunks, vectors)
    assert index.count() == 2

    hits = index.search(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "d#0000"
    assert hits[0].score > hits[1].score


def test_get_round_trips_payload(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    index.upsert([_chunk(3)], np.array([[0, 0, 1, 0]], dtype=np.float32))
    assert index.get("d#0003") == _chunk(3)
    assert index.get("d#9999") is None


def test_recreate_empties_the_collection(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    index.upsert([_chunk(0)], np.array([[1, 0, 0, 0]], dtype=np.float32))
    index.recreate()
    assert index.count() == 0
