import pytest

from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.runner import EvalRun, run_eval, validate_golden_against_corpus
from legalrag.golden import GoldenQuestion, GoldenSetError
from legalrag.models import Chunk, Retrieved

CORPUS_IDS = {"d#0000", "d#0001", "d#0002"}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, k=10):
        return [
            Retrieved(chunk=Chunk(cid, "d", i, cid, cid, {}), score=1.0 - i * 0.1, rank=i + 1)
            for i, cid in enumerate(self.ids[:k])
        ]


def _question(qid="q001", relevant=("d#0000",)):
    return GoldenQuestion(qid, "q", tuple(relevant), "single_clause", True, None)


def test_validate_raises_on_missing_chunk_id():
    question = _question(relevant=("d#9999",))
    with pytest.raises(GoldenSetError, match="d#9999"):
        validate_golden_against_corpus([question], CORPUS_IDS)


def test_validate_skips_unanswerable_questions():
    question = GoldenQuestion("q002", "q", (), "unanswerable", False, None)
    validate_golden_against_corpus([question], CORPUS_IDS)


def test_run_eval_scores_a_perfect_retriever():
    run = run_eval(
        FakeRetriever(["d#0000", "d#0002", "d#0001"]), [_question()], config="dense", ks=[5]
    )
    assert isinstance(run, EvalRun)
    assert run.metrics.recall[5] == 1.0
    assert run.metrics.mrr == 1.0
    assert run.results[0].hit is True


def test_run_eval_scores_a_miss():
    run = run_eval(FakeRetriever(["d#0001"]), [_question()], config="dense", ks=[5])
    assert run.metrics.recall[5] == 0.0
    assert run.results[0].hit is False


def test_run_eval_on_no_questions_returns_zero_metrics():
    run = run_eval(FakeRetriever([]), [], config="dense", ks=[5])
    assert run.metrics.mrr == 0.0
    assert run.results == ()


def test_run_eval_rejects_a_k_max_below_the_deepest_reported_k():
    # Otherwise recall@20 would be computed over five results and still be
    # reported as recall@20.
    with pytest.raises(ValueError, match="k_max"):
        run_eval(
            FakeRetriever(["d#0000"]),
            [_question()],
            config="dense",
            k_max=5,
            ks=[5, 10, 20],
        )


def test_run_eval_reports_only_the_depths_it_can_reach():
    run = run_eval(
        FakeRetriever(["d#0000"]), [_question()], config="dense", k_max=5, ks=[5]
    )
    assert set(run.metrics.recall) == {5}
