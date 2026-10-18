"""Retrieval metrics.

Relevance is binary: a chunk either contains the answer or it does not. Every
function returns 0.0 rather than NaN or a ZeroDivisionError on a degenerate
input, because a metric that raises on an empty result is a metric that hides
a retrieval failure.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def _hits(ranked: Sequence[str], relevant: Collection[str], k: int) -> int:
    return sum(1 for chunk_id in ranked[:k] if chunk_id in relevant)


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top k."""
    if not relevant:
        return 0.0
    return _hits(ranked, relevant, k) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the top k that is relevant. Divides by k, not by the relevant count."""
    if k <= 0:
        return 0.0
    return _hits(ranked, relevant, k) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """1 / position of the first relevant chunk, or 0.0 when there is none."""
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Discounted cumulative gain over the ideal ordering, so several relevant hits all count."""
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant), k) + 1)
    )
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    ndcg: dict[int, float]


def aggregate(
    ranked_lists: Sequence[Sequence[str]],
    relevant_sets: Sequence[Collection[str]],
    ks: Sequence[int],
) -> RetrievalMetrics:
    """Per-question metrics, then the arithmetic mean across questions."""

    def mean(values) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    pairs = list(zip(ranked_lists, relevant_sets))
    return RetrievalMetrics(
        recall={k: mean(recall_at_k(r, rel, k) for r, rel in pairs) for k in ks},
        precision={k: mean(precision_at_k(r, rel, k) for r, rel in pairs) for k in ks},
        mrr=mean(reciprocal_rank(r, rel) for r, rel in pairs),
        ndcg={k: mean(ndcg_at_k(r, rel, k) for r, rel in pairs) for k in ks},
    )
