"""Grading generated answers.

Nothing here calls a model: the judge is faked, and the same client interface
the answer layer uses is reused.
"""

import json

import pytest

from legalrag.answer import LLMError
from legalrag.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_REASONING,
    GradedAnswer,
    Verdict,
    VerdictError,
    build_judge_messages,
    judge_answer,
    parse_verdict,
    render_report,
    retry_once_on_rate_limit,
    summarise,
)
from legalrag.models import Chunk, Retrieved

VALID = json.dumps(
    {
        "correct": "yes",
        "grounded": "yes",
        "refused": False,
        "reason": "States the 90 percent figure and cites the passage containing it.",
    }
)


def _hit(chunk_id: str, text: str, rank: int = 1) -> Retrieved:
    return Retrieved(
        chunk=Chunk(chunk_id, "rbi:1", rank - 1, text, text, {}),
        score=1.0,
        rank=rank,
    )


class FakeClient:
    def __init__(self, reply: str = VALID) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


def _graded(
    qid: str,
    answerable: bool,
    verdict: Verdict | None = None,
    problem: str = "",
) -> GradedAnswer:
    return GradedAnswer(
        qid=qid,
        question="a question",
        category="single_clause",
        answerable=answerable,
        answer_text="an answer",
        cited_chunk_ids=(),
        verdict=verdict,
        problem=problem,
    )


# --- reading the verdict ---------------------------------------------------


def test_parses_a_plain_json_verdict():
    assert parse_verdict(VALID) == Verdict(
        "yes", "yes", False, "States the 90 percent figure and cites the passage containing it."
    )


def test_parses_a_fenced_json_verdict():
    assert parse_verdict(f"```json\n{VALID}\n```").correct == "yes"


def test_parses_json_surrounded_by_chatter():
    assert parse_verdict(f"Here is my verdict:\n{VALID}\nHope that helps.").correct == "yes"


def test_case_and_whitespace_are_normalised():
    reply = json.dumps(
        {"correct": " Yes ", "grounded": "N/A", "refused": False, "reason": "r"}
    )
    verdict = parse_verdict(reply)
    assert verdict.correct == "yes"
    assert verdict.grounded == "n/a"


def test_a_truthy_integer_is_accepted_for_refused():
    reply = json.dumps({"correct": "no", "grounded": "no", "refused": 1, "reason": "r"})
    assert parse_verdict(reply).refused is True


def test_a_missing_field_is_rejected():
    with pytest.raises(VerdictError, match="refused"):
        parse_verdict(json.dumps({"correct": "yes", "grounded": "yes", "reason": "r"}))


def test_an_unknown_grade_is_rejected():
    reply = json.dumps(
        {"correct": "maybe", "grounded": "yes", "refused": False, "reason": "r"}
    )
    with pytest.raises(VerdictError, match="maybe"):
        parse_verdict(reply)


def test_a_reply_that_is_not_json_is_rejected():
    with pytest.raises(VerdictError):
        parse_verdict("I think the answer is probably fine.")


def test_a_reason_that_is_missing_still_yields_a_verdict():
    reply = json.dumps({"correct": "yes", "grounded": "yes", "refused": False})
    assert parse_verdict(reply).reason


# --- what the judge is asked ----------------------------------------------


def test_the_judge_sees_the_question_the_answer_and_the_passages():
    messages = build_judge_messages(
        "what is the rate?", "It is 4% [1].", [_hit("rbi:1#0000", "the rate is 4%")]
    )
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "what is the rate?" in user
    assert "It is 4% [1]." in user
    assert "[1] rbi:1#0000" in user
    assert "the rate is 4%" in user


def test_the_judge_is_asked_for_both_marks_and_a_refusal_flag():
    system = build_judge_messages("q", "a", [_hit("rbi:1#0000", "t")])[0]["content"].lower()
    assert "json" in system
    assert "correct" in system
    assert "grounded" in system
    assert "refused" in system
    assert "outside" in system


def test_the_judge_reply_fits_the_providers_output_token_limit():
    """Qwen 3.8 on the free tier meters output at 1000 tokens a minute and
    rejects a request whose estimated output goes over. Six of twenty verdicts
    were cut off at the answerer's budget, and a larger one was refused."""
    assert JUDGE_MAX_TOKENS <= 1000


def test_the_judge_reasons_silently():
    """Reasoning tokens count as output on that meter, so the judge turns it off
    at the client and carries the accountability in its written reason instead."""
    assert JUDGE_REASONING == "none"


def test_judge_answer_parses_what_the_client_returns():
    client = FakeClient()
    verdict = judge_answer("q", "a", [_hit("rbi:1#0000", "t")], client)
    assert verdict.correct == "yes"
    assert "Question:" in client.calls[0][1]["content"]


# --- surviving the free tier's rate limit ---------------------------------


def test_a_rate_limited_call_is_retried_once_after_a_wait():
    slept: list[float] = []
    calls = {"n": 0}

    class Flaky:
        def complete(self, messages: list[dict]) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError("the model returned 429: Rate limit reached")
            return VALID

    verdict = retry_once_on_rate_limit(
        lambda: judge_answer("q", "a", [_hit("rbi:1#0000", "t")], Flaky()),
        sleep=slept.append,
    )
    assert verdict.correct == "yes"
    assert slept == [30.0]


def test_an_unrelated_failure_is_not_retried():
    slept: list[float] = []

    class Broken:
        def complete(self, messages: list[dict]) -> str:
            raise LLMError("the model returned 401: invalid api key")

    with pytest.raises(LLMError, match="401"):
        retry_once_on_rate_limit(
            lambda: judge_answer("q", "a", [_hit("rbi:1#0000", "t")], Broken()),
            sleep=slept.append,
        )
    assert slept == []


def test_a_second_rate_limit_is_given_up_on():
    calls = {"n": 0}

    class Always:
        def complete(self, messages: list[dict]) -> str:
            calls["n"] += 1
            raise LLMError("the model returned 429: Rate limit reached")

    with pytest.raises(LLMError, match="429"):
        retry_once_on_rate_limit(
            lambda: judge_answer("q", "a", [_hit("rbi:1#0000", "t")], Always()),
            sleep=lambda seconds: None,
        )
    assert calls["n"] == 2


# --- counting --------------------------------------------------------------


def test_summary_counts_correct_answers_and_correct_refusals():
    summary = summarise(
        [
            _graded("q1", True, Verdict("yes", "yes", False, "r")),
            _graded("q2", True, Verdict("no", "yes", True, "r")),
            _graded("q3", False, Verdict("no", "n/a", True, "r")),
            _graded("q4", False, Verdict("yes", "yes", False, "r")),
        ]
    )
    assert summary["answerable"] == 2
    assert summary["answered_correctly"] == 1
    assert summary["unanswerable"] == 2
    assert summary["refused_correctly"] == 1
    assert summary["answered_anyway"] == 1


def test_summary_separates_a_partial_answer_from_a_wrong_one():
    summary = summarise(
        [
            _graded("q1", True, Verdict("yes", "yes", False, "r")),
            _graded("q2", True, Verdict("partly", "yes", False, "r")),
            _graded("q3", True, Verdict("no", "yes", False, "r")),
        ]
    )
    assert summary["answerable"] == 3
    assert summary["answered_correctly"] == 1
    assert summary["answered_partly"] == 1


def test_summary_counts_partly_supported_answers_separately():
    """'Supported in part' is not 'invented'. But a headline of zero must not let
    the weaker one hide: some claims in those answers were not supported."""
    summary = summarise(
        [
            _graded("q1", True, Verdict("yes", "partly", False, "r")),
            _graded("q2", True, Verdict("yes", "no", False, "r")),
            _graded("q3", True, Verdict("yes", "yes", False, "r")),
        ]
    )
    assert summary["judged"] == 3
    assert summary["partly_supported"] == 1
    assert summary["unsupported"] == 1


def test_invented_content_is_counted_on_its_own():
    """A fluent, well-cited, wrong answer is the failure worth a number of its own."""
    summary = summarise(
        [
            _graded("q1", True, Verdict("yes", "no", False, "r")),
            _graded("q2", True, Verdict("yes", "partly", False, "r")),
            _graded("q3", True, Verdict("yes", "yes", False, "r")),
        ]
    )
    assert summary["answered_correctly"] == 3
    assert summary["unsupported"] == 1


def test_an_unjudged_question_is_kept_out_of_the_denominator():
    summary = summarise(
        [
            _graded("q1", True, None, "rate limit reached"),
            _graded("q2", True, Verdict("no", "yes", False, "r")),
        ]
    )
    assert summary["unjudged"] == 1
    assert summary["answerable"] == 1
    assert summary["answered_correctly"] == 0


# --- the report ------------------------------------------------------------


def _report(graded):
    return render_report(graded, summarise(graded), model="a-judge-model")


def test_report_leads_with_the_summary_and_covers_every_question():
    report = _report(
        [
            _graded("q1", True, Verdict("yes", "yes", False, "States the rate.")),
            _graded("q2", False, Verdict("no", "n/a", True, "Outside the corpus.")),
        ]
    )
    assert report.startswith("#")
    assert "1 of 1 answerable questions answered correctly" in report
    assert "1 of 1 unanswerable questions correctly refused" in report
    assert "`q1`" in report
    assert "`q2`" in report
    assert "States the rate." in report
    assert "Outside the corpus." in report


def test_report_does_not_let_partly_supported_hide_behind_a_zero():
    """A headline that reports zero unsupported answers while three were only
    partly supported reads as a clean bill of health. It is not one."""
    report = _report([_graded("q1", True, Verdict("yes", "partly", False, "r"))])
    assert "supported only in part" in report


def test_report_says_the_judge_had_no_thinking_budget():
    """A report that hides how deep the judging went is the problem this project
    exists to avoid, and the report is what gets shared."""
    report = _report([_graded("q1", True, Verdict("yes", "yes", False, "r"))])
    assert "reasoning switched off" in report


def test_report_names_the_model_that_did_the_judging():
    assert "a-judge-model" in _report([_graded("q1", True, Verdict("yes", "yes", False, "r"))])


def test_report_says_when_a_question_was_not_judged():
    report = _report([_graded("q1", True, None, "rate limit reached")])
    assert "rate limit reached" in report
    assert "not judged" in report


def test_report_states_that_a_model_judged_a_model():
    report = _report([_graded("q1", True, Verdict("yes", "yes", False, "r"))])
    assert "second opinion" in report.lower() or "not proof" in report.lower()
