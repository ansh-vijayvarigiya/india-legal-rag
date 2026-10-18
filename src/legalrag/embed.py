"""Local embedding wrapper. No embedding API, so no key and no per-token cost."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _preferred_device() -> str:
    # Blackwell needs a recent PyTorch with CUDA 12.8 wheels; CPU is fine at this
    # corpus size, so a missing CUDA build degrades rather than fails.
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """A `sentence-transformers` model, loaded once, returning unit-length vectors."""

    def __init__(
        self, model_name: str, *, device: str | None = None, batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or _preferred_device()
        self._model = SentenceTransformer(model_name, device=self.device)

    @property
    def dim(self) -> int:
        # sentence-transformers renamed this accessor; keep both so the >=3.0 pin holds.
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return getter()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
