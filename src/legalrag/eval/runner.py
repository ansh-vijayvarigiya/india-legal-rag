"""The evaluation runner: one EvalRun per retrieval configuration."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from legalrag.eval.metrics import RetrievalMetrics, aggregate
from legalrag.golden import GoldenQuestion, GoldenSetError
from legalrag.retrieve import Retriever

DEFAULT_K_MAX = 20
DEFAULT_KS = (5, 10, 20)


@dataclass(frozen=True, slots=True)
class QuestionResult:
    qid: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    hit: bool


@dataclass(frozen=True, slots=True)
class EvalRun:
    config: str
    ks: tuple[int, ...]
    metrics: RetrievalMetrics
    results: tuple[QuestionResult, ...]


def validate_golden_against_corpus(
    questions: Sequence[GoldenQuestion], known_chunk_ids: Collection[str]
) -> None:
    """Fail loudly when a label points at a chunk the corpus does not contain.

    Without this, a corpus rebuild that renumbers chunks would score zero and
    look like a retrieval regression. Unanswerable questions have no ids to check.
    """
    for question in questions:
        for chunk_id in question.relevant_chunk_ids:
            if chunk_id not in known_chunk_ids:
                raise GoldenSetError(
                    f"{question.qid}: relevant chunk {chunk_id!r} is not in the corpus"
                )


def run_eval(
    retriever: Retriever,
    questions: Sequence[GoldenQuestion],
    *,
    config: str,
    k_max: int = DEFAULT_K_MAX,
    ks: Sequence[int] = DEFAULT_KS,
) -> EvalRun:
    """Retrieve `k_max` results per question and score them.

    The corpus check lives in the caller, which is the layer that holds the
    chunk store; `Retriever` exposes no way to enumerate the corpus.
    """
    results: list[QuestionResult] = []
    for question in questions:
        hits = retriever.retrieve(question.question, k=k_max)
        retrieved_ids = tuple(hit.chunk.chunk_id for hit in hits)
        relevant_ids = tuple(question.relevant_chunk_ids)
        results.append(
            QuestionResult(
                qid=question.qid,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                hit=bool(set(retrieved_ids) & set(relevant_ids)),
            )
        )

    return EvalRun(
        config=config,
        ks=tuple(ks),
        metrics=aggregate(
            [result.retrieved_ids for result in results],
            [set(result.relevant_ids) for result in results],
            ks,
        ),
        results=tuple(results),
    )
