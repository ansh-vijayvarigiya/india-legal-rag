"""Grading the answers that `legalrag.answer` writes.

Retrieval is scored by `legalrag.eval`, and that measurement stops short of the
thing a reader actually sees. An answer can be fluent, well cited, and wrong,
and no retrieval metric can tell the difference.

Two marks are kept apart on purpose: whether the answer resolves the question,
and whether every claim in it is supported by the passages it was given. A
single score would let one invented figure be averaged away by three correct
sentences, and an invented figure is the failure that matters in regulatory
text.

An AI grading an AI is a second opinion, not ground truth, and the report says
so in its own words. The mitigations are that the judge is a different model
from the writer, that it must give a reason for every mark, and that the
passages behind every reason are in this repository.

Run with `python -m legalrag.judge`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from legalrag.answer import (
    MAX_PASSAGES,
    ChatClient,
    LLMError,
    OpenAIChatClient,
    answer,
    is_rate_limit,
    render_context,
)
from legalrag.config import Settings
from legalrag.corpus import read_chunks
from legalrag.embed import Embedder
from legalrag.golden import GoldenSetError, load_golden_set
from legalrag.index import ChunkIndex
from legalrag.models import Retrieved
from legalrag.retrieve import DenseRetriever

CORRECT_VALUES = ("yes", "partly", "no")
GROUNDED_VALUES = ("yes", "partly", "no", "n/a")

# Long enough for a transient quota to refill, short enough not to stall a batch.
RETRY_WAIT_SECONDS = 30.0
# Free tiers meter tokens per minute, and one question costs two calls across two
# models. The wait keeps a whole run inside the allowance instead of recovering
# from it one question at a time.
DEFAULT_QUESTION_DELAY = 20.0
# How many passages the answer was written from, so the judge sees a fair set.
JUDGE_PASSAGES = MAX_PASSAGES
# Qwen 3.8 on the free tier meters output tokens at a thousand a minute and
# rejects a request whose estimated output goes over it, so a big budget is not
# available even in principle. Six of twenty verdicts were cut off at the
# answerer's budget of 700 before reasoning was turned off; with it off, a
# verdict needs a fraction of this.
JUDGE_MAX_TOKENS = 400
# Qwen 3.8 turns reasoning on with `low` and off with `none`. Reasoning tokens
# count against that same output meter, and `none` keeps a whole run inside it
# instead of losing questions to a cut-off reply. The accountability that would
# have been in the reasoning lives in the written reason for each mark.
JUDGE_REASONING = "none"

JUDGE_INSTRUCTIONS = """You grade an answer that was written from numbered passages.

You are given the question, the passages the answer was written from, and the
answer itself. The passages are the only source of truth available to you. Do
not use knowledge from outside them when deciding whether the answer is right.

Reply with a single JSON object and nothing else, in exactly this shape:

{"correct": "yes|partly|no", "grounded": "yes|partly|no|n/a", "refused": true|false, "reason": "one sentence"}

What each field means:

- "correct": does the answer resolve what the question asked? "yes" when a
  reader gets what they asked for, "partly" when it answers only some of it or
  stays vague, "no" when it does not answer it.
- "grounded": is every factual claim in the answer supported by the passages?
  "yes" when all of it is, "partly" when some claims are not, "no" when the
  answer states a figure, date, threshold or rule that the passages do not
  contain. Use "n/a" when the answer declines to answer and makes no claims.
- "refused": true when the answer declines because the passages do not contain
  the answer. A refusal is not a failure by itself, and refusing is the correct
  behaviour when the passages genuinely do not answer the question.
- "reason": one sentence naming the specific thing the marks are based on.

Judge the answer, not the passages. Do not give credit for the passages being
correct, and do not penalise an answer for the passages being incomplete.
"""

_CAVEAT = (
    "The judge is a language model, so these marks are a second opinion rather "
    "than proof, and it shares some of the blind spots of the model it is "
    "judging. It is a different model from the one that wrote the answers, it "
    "must give a reason for every mark, and every reason can be checked by hand "
    "against passages that are in this repository."
)


class VerdictError(ValueError):
    """The judge's reply could not be read as a verdict."""


@dataclass(frozen=True, slots=True)
class Verdict:
    correct: str
    grounded: str
    refused: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GradedAnswer:
    qid: str
    question: str
    category: str
    answerable: bool
    answer_text: str
    cited_chunk_ids: tuple[str, ...]
    verdict: Verdict | None
    problem: str = ""


def _extract_json(reply: str) -> dict:
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end < start:
        raise VerdictError(f"the judge did not reply with JSON: {reply[:120]!r}")
    try:
        payload = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        raise VerdictError(f"the judge's JSON did not parse: {error}") from error
    if not isinstance(payload, dict):
        raise VerdictError("the judge's JSON was not an object")
    return payload


def _grade(payload: dict, field: str, allowed: Sequence[str]) -> str:
    if field not in payload:
        raise VerdictError(f"the judge's reply has no {field!r} key")
    value = str(payload[field]).strip().lower()
    if value not in allowed:
        raise VerdictError(
            f"{field} was {value!r}, expected one of {', '.join(allowed)}"
        )
    return value


def _flag(payload: dict, field: str) -> bool:
    if field not in payload:
        raise VerdictError(f"the judge's reply has no {field!r} key")
    value = payload[field]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, int):
        return bool(value)
    raise VerdictError(f"{field} was {value!r}, expected true or false")


def parse_verdict(reply: str) -> Verdict:
    """Read a verdict out of the judge's reply, tolerating fences and chatter."""
    payload = _extract_json(reply)
    reason = str(payload.get("reason", "")).strip() or "(the judge gave no reason)"
    return Verdict(
        correct=_grade(payload, "correct", CORRECT_VALUES),
        grounded=_grade(payload, "grounded", GROUNDED_VALUES),
        refused=_flag(payload, "refused"),
        reason=reason,
    )


def build_judge_messages(
    question: str, answer_text: str, hits: Sequence[Retrieved]
) -> list[dict]:
    user = (
        f"Question: {question}\n\n"
        f"Passages the answer was written from:\n\n{render_context(hits)}\n\n"
        f"Answer:\n{answer_text}"
    )
    return [
        {"role": "system", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]


def judge_answer(
    question: str, answer_text: str, hits: Sequence[Retrieved], client: ChatClient
) -> Verdict:
    return parse_verdict(client.complete(build_judge_messages(question, answer_text, hits)))


T = TypeVar("T")


def retry_once_on_rate_limit(
    operation: Callable[[], T],
    *,
    sleep: Callable[[float], None],
    wait: float = RETRY_WAIT_SECONDS,
) -> T:
    """Run `operation`, waiting out one rate limit. Other failures pass through."""
    try:
        return operation()
    except LLMError as error:
        if not is_rate_limit(error):
            raise
        sleep(wait)
        return operation()


def summarise(graded: Sequence[GradedAnswer]) -> dict[str, int]:
    """Counts over the questions that were actually judged.

    An unjudged question is a failure of this harness, not of the answer, so it
    is counted separately rather than left in a denominator to look like a miss.
    """
    judged = [item for item in graded if item.verdict is not None]
    answerable = [item for item in judged if item.answerable]
    unanswerable = [item for item in judged if not item.answerable]
    return {
        "judged": len(judged),
        "answerable": len(answerable),
        "answered_correctly": sum(
            1 for item in answerable if item.verdict.correct == "yes"
        ),
        "answered_partly": sum(
            1 for item in answerable if item.verdict.correct == "partly"
        ),
        "unanswerable": len(unanswerable),
        "refused_correctly": sum(1 for item in unanswerable if item.verdict.refused),
        "answered_anyway": sum(1 for item in unanswerable if not item.verdict.refused),
        "unsupported": sum(1 for item in judged if item.verdict.grounded == "no"),
        "partly_supported": sum(
            1 for item in judged if item.verdict.grounded == "partly"
        ),
        "unjudged": len(graded) - len(judged),
    }


def render_report(
    graded: Sequence[GradedAnswer], summary: dict[str, int], *, model: str
) -> str:
    lines = [
        "# Answer evaluation",
        "",
        f"Judged by `{model}`, a different model from the one that wrote the "
        f"answers, running with reasoning switched off: the free tier meters "
        f"output tokens at a thousand a minute, and hidden reasoning counts as "
        f"output. What a judge could find with a thinking budget is in the "
        f"README, and it is not nothing.",
        "",
        f"**{summary['answered_correctly']} of {summary['answerable']} answerable "
        f"questions answered correctly, {summary['answered_partly']} partly. "
        f"{summary['refused_correctly']} of "
        f"{summary['unanswerable']} unanswerable questions correctly refused. "
        f"Of {summary['judged']} answers judged, {summary['unsupported']} stated "
        f"something the passages do not support and "
        f"{summary['partly_supported']} were supported only in part.**",
        "",
    ]
    if summary["unjudged"]:
        lines += [
            f"{summary['unjudged']} question(s) were not judged. They are kept out "
            "of the counts above rather than scored, because a failure of this "
            "harness is not a failure of the answer.",
            "",
        ]

    lines += ["| qid | category | right? | backed up? | refused? |", "| --- | --- | --- | --- | --- |"]
    for item in graded:
        if item.verdict is None:
            lines.append(f"| `{item.qid}` | {item.category} | not judged | | |")
        else:
            lines.append(
                f"| `{item.qid}` | {item.category} | {item.verdict.correct} | "
                f"{item.verdict.grounded} | {'yes' if item.verdict.refused else 'no'} |"
            )
    lines += [""]

    for item in graded:
        lines += [f"### {item.qid} · {item.category}", "", f"**Question.** {item.question}", ""]
        lines += [f"**Answer.** {item.answer_text}", ""]
        if item.verdict is None:
            lines += [f"**Verdict.** not judged — {item.problem}", ""]
        else:
            verdict = item.verdict
            lines += [
                f"**Verdict.** right: **{verdict.correct}** · backed up: "
                f"**{verdict.grounded}** · refused: "
                f"**{'yes' if verdict.refused else 'no'}**",
                "",
                f"**Why the judge says so.** {verdict.reason}",
                "",
            ]

    lines += [_CAVEAT, ""]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="legalrag.judge")
    parser.add_argument("--golden", type=Path, default=None, help="golden set YAML")
    parser.add_argument("--out", type=Path, default=Path("answers.md"))
    parser.add_argument(
        "--limit", type=int, default=None, help="judge only the first N questions"
    )
    parser.add_argument("--delay", type=float, default=DEFAULT_QUESTION_DELAY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings()

    key = settings.llm_api_key.get_secret_value()
    if not key:
        print(
            "error: LEGALRAG_LLM_API_KEY is not set, so answers cannot be written "
            "or judged. See the README.",
            file=sys.stderr,
        )
        return 1

    golden_path = args.golden or (settings.data_dir / "golden" / "milestone1.yaml")
    chunks_path = settings.data_dir / "chunks" / "chunks.jsonl"
    chunks = list(read_chunks(chunks_path))
    if not chunks:
        print(f"error: no chunks at {chunks_path}, ingest first", file=sys.stderr)
        return 1
    try:
        questions = load_golden_set(golden_path)
    except (FileNotFoundError, GoldenSetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not questions:
        print(f"error: golden set at {golden_path} has no questions", file=sys.stderr)
        return 1
    if args.limit is not None:
        questions = questions[: args.limit]

    embedder = Embedder(settings.embedding_model)
    index = ChunkIndex(settings.qdrant_path, settings.collection, embedder.dim)
    index.recreate()
    index.upsert(chunks, embedder.encode([chunk.embed_text for chunk in chunks]))
    retriever = DenseRetriever(index, embedder)

    writer = OpenAIChatClient(settings.llm_base_url, key, settings.llm_model)
    judge = OpenAIChatClient(
        settings.llm_base_url,
        key,
        settings.judge_model,
        max_tokens=JUDGE_MAX_TOKENS,
        reasoning_effort=JUDGE_REASONING,
    )

    graded: list[GradedAnswer] = []
    for position, question in enumerate(questions):
        if position:
            time.sleep(args.delay)
        hits = retriever.retrieve(question.question, k=JUDGE_PASSAGES)
        try:
            written = retry_once_on_rate_limit(
                lambda: answer(question.question, hits, writer), sleep=time.sleep
            )
        except LLMError as error:
            graded.append(
                GradedAnswer(
                    qid=question.qid,
                    question=question.question,
                    category=question.category,
                    answerable=question.answerable,
                    answer_text="",
                    cited_chunk_ids=(),
                    verdict=None,
                    problem=f"answering failed: {error}",
                )
            )
            print(f"{question.qid}: answering failed: {error}")
            continue

        try:
            verdict = retry_once_on_rate_limit(
                lambda: judge_answer(question.question, written.text, hits, judge),
                sleep=time.sleep,
            )
            problem = ""
        except (LLMError, VerdictError) as error:
            verdict = None
            problem = f"judging failed: {error}"

        graded.append(
            GradedAnswer(
                qid=question.qid,
                question=question.question,
                category=question.category,
                answerable=question.answerable,
                answer_text=written.text,
                cited_chunk_ids=written.used_chunk_ids,
                verdict=verdict,
                problem=problem,
            )
        )
        mark = "not judged" if verdict is None else f"{verdict.correct}/{verdict.grounded}"
        print(f"{question.qid}: {mark}")

    summary = summarise(graded)
    args.out.write_text(
        render_report(graded, summary, model=settings.judge_model), encoding="utf-8"
    )
    print(
        f"\n{summary['answered_correctly']} of {summary['answerable']} answerable "
        f"questions answered correctly ({summary['answered_partly']} partly); "
        f"{summary['refused_correctly']} of {summary['unanswerable']} unanswerable "
        f"questions correctly refused; {summary['unsupported']} unsupported claims, "
        f"{summary['partly_supported']} supported only in part."
    )
    print(f"wrote {args.out}")
    index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
