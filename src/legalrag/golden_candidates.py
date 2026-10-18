"""Input for golden-set candidate generation.

Nothing here calls a model. `render_candidates` produces the text for a language
model session that proposes one question per chunk, and a human then verifies
every proposal against the chunk before it enters the golden set.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from legalrag.models import Chunk


def sample_chunks(chunks: Sequence[Chunk], n: int, *, seed: int = 0) -> list[Chunk]:
    """A seeded sample, so a candidate sheet can be reproduced exactly."""
    if n >= len(chunks):
        return list(chunks)
    return random.Random(seed).sample(list(chunks), n)


def render_candidates(chunks: Sequence[Chunk]) -> str:
    """One fenced block per chunk, ready to paste into a model session."""
    blocks = [
        f"### {chunk.chunk_id}\n"
        f"citation: {chunk.metadata.get('citation', '')}\n"
        f"```\n{chunk.text}\n```"
        for chunk in chunks
    ]
    return "\n\n".join(blocks)
