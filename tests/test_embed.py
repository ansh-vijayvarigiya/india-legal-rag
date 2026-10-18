import numpy as np
import pytest

from legalrag.embed import Embedder

pytestmark = pytest.mark.slow


def test_encode_returns_normalised_vectors():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    vectors = embedder.encode(["payment of tax", "input tax credit"])
    assert vectors.shape == (2, embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_is_deterministic():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    first = embedder.encode(["same text"])
    second = embedder.encode(["same text"])
    assert np.allclose(first, second)


def test_encode_empty_returns_zero_rows():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    assert embedder.encode([]).shape == (0, embedder.dim)
