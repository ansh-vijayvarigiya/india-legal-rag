"""The vector index. Every Qdrant call lives in this module.

Qdrant runs in local on-disk mode because Docker is unavailable. Nothing here
leaks Qdrant types, so retrieval code could move to a Qdrant server unchanged.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from legalrag.models import Chunk, Retrieved


def _point_id(chunk_id: str) -> str:
    """Qdrant accepts only unsigned integers or UUIDs, so map the chunk id onto a UUID.

    uuid5 is deterministic, so the mapping survives re-indexing.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class ChunkIndex:
    def __init__(self, path: Path, collection: str, dim: int) -> None:
        self.path = path
        self.collection = collection
        self.dim = dim
        self._client = QdrantClient(path=str(path))

    def recreate(self) -> None:
        """Start from an empty collection.

        Qdrant local mode does not persist a collection deletion, so points
        survive `delete_collection` and even a reopened client still sees them.
        Dropping the storage directory is the only way to empty the index.
        """
        self.close()
        shutil.rmtree(self.path, ignore_errors=True)
        self._client = QdrantClient(path=str(self.path))
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def close(self) -> None:
        """Release the storage lock. Local mode holds it for the client's lifetime."""
        self._client.close()

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=[float(value) for value in vector],
                payload=chunk.to_json(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: np.ndarray, k: int) -> list[Retrieved]:
        response = self._client.query_points(
            collection_name=self.collection,
            query=[float(value) for value in vector],
            limit=k,
            with_payload=True,
        )
        return [
            Retrieved(
                chunk=Chunk.from_json(point.payload),
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        records = self._client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(chunk_id)],
            with_payload=True,
        )
        if not records:
            return None
        return Chunk.from_json(records[0].payload)

    def count(self) -> int:
        return self._client.get_collection(self.collection).points_count or 0
