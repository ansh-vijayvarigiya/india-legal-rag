"""Retrieval configurations.

Evaluation code is written against `Retriever`, so a later configuration — hybrid,
reranked, query-transformed — drops in without touching the metrics or the runner.
"""

from __future__ import annotations

from typing import Protocol

from legalrag.embed import Embedder
from legalrag.index import ChunkIndex
from legalrag.models import Retrieved


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]: ...


class DenseRetriever:
    """Configuration 0: dense only. No query transformation, no reranking."""

    def __init__(self, index: ChunkIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:
        if self.index.count() == 0:
            return []
        vectors = self.embedder.encode([query])
        return self.index.search(vectors[0], k)
