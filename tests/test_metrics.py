import math

import pytest

from legalrag.eval.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANT = {"a", "b", "c"}


def test_recall_at_k_counts_a_fraction_of_all_relevant():
    ranked = ["a", "x", "y", "z", "b"]
    assert recall_at_k(ranked, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, RELEVANT, 2) == pytest.approx(1 / 3)


def test_precision_at_k_divides_by_k_not_by_relevant_count():
    assert precision_at_k(["a", "x"], RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(["a", "x", "y", "z", "b"], RELEVANT, 5) == pytest.approx(0.4)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["x", "y", "b"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], RELEVANT) == 1.0


def test_ndcg_at_k_matches_hand_computation():
    # relevant at ranks 1 and 3, three relevant total
    ranked = ["a", "x", "b"]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, RELEVANT, 3) == pytest.approx(dcg / idcg)


@pytest.mark.parametrize(
    "func",
    [
        lambda r: recall_at_k(r, RELEVANT, 5),
        lambda r: precision_at_k(r, RELEVANT, 5),
        lambda r: reciprocal_rank(r, RELEVANT),
        lambda r: ndcg_at_k(r, RELEVANT, 5),
    ],
)
def test_empty_retrieval_scores_zero_without_nan(func):
    value = func([])
    assert value == 0.0
    assert not math.isnan(value)


def test_empty_relevant_set_scores_zero_without_nan():
    for value in (
        recall_at_k(["a"], set(), 5),
        precision_at_k(["a"], set(), 5),
        reciprocal_rank(["a"], set()),
        ndcg_at_k(["a"], set(), 5),
    ):
        assert value == 0.0
        assert not math.isnan(value)


def test_aggregate_averages_across_questions():
    metrics = aggregate([["a", "x"], ["b", "c"]], [{"a"}, {"b", "c"}], ks=[2])
    assert metrics.recall[2] == pytest.approx(1.0)
    assert metrics.precision[2] == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx(1.0)


def test_aggregate_with_no_questions_scores_zero():
    metrics = aggregate([], [], ks=[5])
    assert metrics.recall[5] == 0.0
    assert metrics.precision[5] == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg[5] == 0.0
