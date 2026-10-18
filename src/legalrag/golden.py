"""The golden set: the hand-verified ground truth every reported number rests on.

A wrong label here invalidates the report rather than the code, so the loader
refuses anything it cannot check rather than guessing at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = (
    "single_clause",
    "multi_clause",
    "cross_document",
    "temporal",
    "unanswerable",
)
UNAANSWERABLE = "unanswerable"
REQUIRED_FIELDS = ("qid", "question", "relevant_chunk_ids", "category", "answerable")


class GoldenSetError(ValueError):
    """The golden set file is not valid."""


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    qid: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    category: str
    answerable: bool
    gold_answer: str | None


def _check(entry: dict, index: int) -> str:
    qid = entry.get("qid")
    label = qid if isinstance(qid, str) and qid else f"entry {index}"

    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise GoldenSetError(f"{label}: missing required field {field!r}")

    question = entry["question"]
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"{label}: question must be a non-empty string")

    category = entry["category"]
    if category not in CATEGORIES:
        raise GoldenSetError(
            f"{label}: unknown category {category!r}; expected one of {', '.join(CATEGORIES)}"
        )

    relevant = entry["relevant_chunk_ids"]
    if not isinstance(relevant, list):
        raise GoldenSetError(f"{label}: relevant_chunk_ids must be a list")

    answerable = entry["answerable"]
    if not isinstance(answerable, bool):
        raise GoldenSetError(f"{label}: answerable must be true or false")

    if answerable and not relevant:
        raise GoldenSetError(
            f"{label}: relevant_chunk_ids must not be empty for an answerable question"
        )
    if not answerable and relevant:
        raise GoldenSetError(
            f"{label}: relevant_chunk_ids must be empty for an unanswerable question"
        )
    if category == UNAANSWERABLE and answerable:
        raise GoldenSetError(f"{label}: category {UNAANSWERABLE!r} cannot be answerable")

    return label


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Load and validate the golden set. An empty or comment-only file yields `[]`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise GoldenSetError(f"{path}: expected a YAML list of questions")

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise GoldenSetError(f"entry {index}: expected a mapping")

        label = _check(entry, index)
        qid = entry["qid"]
        if qid in seen:
            raise GoldenSetError(f"{label}: duplicate qid")
        seen.add(qid)

        gold_answer = entry.get("gold_answer")
        questions.append(
            GoldenQuestion(
                qid=qid,
                question=entry["question"],
                relevant_chunk_ids=tuple(str(chunk_id) for chunk_id in entry["relevant_chunk_ids"]),
                category=entry["category"],
                answerable=entry["answerable"],
                gold_answer=None if gold_answer is None else str(gold_answer),
            )
        )

    return questions
