import numpy as np

from legalrag.index import ChunkIndex
from legalrag.models import Chunk
from legalrag.retrieve import DenseRetriever

VECTORS = {"alpha": [1, 0, 0], "beta": [0, 1, 0], "query about alpha": [1, 0, 0]}


class FakeEmbedder:
    dim = 3

    def encode(self, texts):
        return np.array([VECTORS[t] for t in texts], dtype=np.float32)


def _retriever(tmp_path) -> DenseRetriever:
    index = ChunkIndex(tmp_path / "qdrant", "test", 3)
    index.recreate()
    chunks = [
        Chunk("d#0000", "d", 0, "alpha", "alpha", {}),
        Chunk("d#0001", "d", 1, "beta", "beta", {}),
    ]
    index.upsert(chunks, np.array([VECTORS["alpha"], VECTORS["beta"]], dtype=np.float32))
    return DenseRetriever(index, FakeEmbedder())


def test_retrieve_returns_top_k_in_rank_order(tmp_path):
    hits = _retriever(tmp_path).retrieve("query about alpha", k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "d#0000"


def test_retrieve_respects_k(tmp_path):
    assert len(_retriever(tmp_path).retrieve("query about alpha", k=1)) == 1


def test_retrieve_on_empty_index_returns_empty(tmp_path):
    index = ChunkIndex(tmp_path / "empty", "test", 3)
    index.recreate()
    assert DenseRetriever(index, FakeEmbedder()).retrieve("alpha", k=5) == []
