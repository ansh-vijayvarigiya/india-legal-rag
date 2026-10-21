from legalrag.answer import LLMError
from legalrag.config import Settings
from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.runner import EvalRun, QuestionResult
from legalrag.golden import GoldenQuestion
from legalrag.inspector import (
    _answer_entry,
    answers_report,
    example_questions,
    judge_client,
    metrics_rows,
    question_rows,
    rate_limit_message,
    results_rows,
)
from legalrag.models import Chunk, Retrieved


def test_results_rows_shapes_hits_for_display():
    hits = [
        Retrieved(
            chunk=Chunk(
                "d#0000",
                "d",
                0,
                "text zero",
                "text zero",
                {"citation": "CGST Rules, 2017, Rule 36(4)"},
            ),
            score=0.91,
            rank=1,
        ),
        Retrieved(
            chunk=Chunk("d#0001", "d", 1, "text one", "text one", {}),
            score=0.42,
            rank=2,
        ),
    ]
    rows = results_rows(hits)
    assert rows[0]["rank"] == 1
    assert rows[0]["chunk_id"] == "d#0000"
    assert rows[0]["citation"] == "CGST Rules, 2017, Rule 36(4)"
    assert rows[0]["text"] == "text zero"
    assert list(rows[0]) == [
        "rank",
        "score",
        "chunk_id",
        "doc_id",
        "citation",
        "text",
        "chars",
    ]


def test_results_rows_handles_no_hits():
    assert results_rows([]) == []


def test_results_rows_uses_empty_citation_when_metadata_is_missing():
    hits = [Retrieved(chunk=Chunk("d#0000", "d", 0, "t", "t", {}), score=1.0, rank=1)]
    assert results_rows(hits)[0]["citation"] == ""


GOLDEN = (
    "- qid: q001\n"
    "  question: What is the bank rate?\n"
    "  relevant_chunk_ids: ['a#0000']\n"
    "  category: single_clause\n"
    "  answerable: true\n"
    "- qid: q002\n"
    "  question: What is the GST rate on unicorn feed?\n"
    "  relevant_chunk_ids: []\n"
    "  category: unanswerable\n"
    "  answerable: false\n"
)


def test_example_questions_come_from_the_golden_set(tmp_path):
    """Clickable examples, so nobody has to invent a question to try."""
    path = tmp_path / "golden.yaml"
    path.write_text(GOLDEN, encoding="utf-8")
    assert example_questions(path, 4) == ["What is the bank rate?"]


def test_example_questions_stop_at_the_requested_count(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text(
        GOLDEN + "- qid: q003\n  question: What is the CRR?\n"
        "  relevant_chunk_ids: ['a#0001']\n  category: single_clause\n  answerable: true\n",
        encoding="utf-8",
    )
    assert len(example_questions(path, 1)) == 1


def test_example_questions_are_empty_when_the_golden_set_is_missing(tmp_path):
    assert example_questions(tmp_path / "absent.yaml", 4) == []


def test_a_rate_limit_is_explained_in_plain_words():
    message = rate_limit_message(
        LLMError("the model returned 429: Rate limit reached for model on tokens per minute")
    )
    assert "free" in message.lower()
    assert "429" not in message


def test_other_errors_are_shown_as_they_arrived():
    plain = "the model returned 401: invalid api key"
    assert rate_limit_message(LLMError(plain)) == plain


# --- the scoreboard -------------------------------------------------------


def _question(
    qid: str,
    category: str = "single_clause",
    relevant: tuple[str, ...] = ("a#0000",),
    answerable: bool = True,
    text: str = "a question",
) -> GoldenQuestion:
    return GoldenQuestion(qid, text, relevant, category, answerable, None)


def _run(results: list[QuestionResult]) -> EvalRun:
    return EvalRun(
        config="dense-fixed",
        ks=(5, 10, 20),
        metrics=RetrievalMetrics(
            recall={5: 0.78, 10: 0.82, 20: 0.85},
            precision={5: 0.22},
            mrr=0.73,
            ndcg={10: 0.73},
        ),
        results=tuple(results),
    )


def test_metrics_rows_report_the_headline_numbers():
    rows = metrics_rows(_run([]).metrics)
    assert [row["metric"] for row in rows] == [
        "recall@5",
        "recall@10",
        "recall@20",
        "precision@5",
        "mrr",
        "ndcg@10",
    ]
    assert rows[0]["score"] == 0.78
    assert rows[4]["score"] == 0.73


def test_every_metric_says_what_it_means_in_words():
    """The number alone is the problem this table exists to fix: 0.78 means
    nothing to a reader who has not met recall before."""
    for row in metrics_rows(_run([]).metrics):
        meaning = row["what it means"]
        assert meaning and len(meaning.split()) >= 5
        assert row["metric"] not in meaning


def test_question_rows_show_the_question_rather_than_chunk_ids():
    run = _run([QuestionResult("q001", ("a#0009",), ("a#0009",), True)])
    row = question_rows(run, [_question("q001", text="What is the Bank Rate?")])[0]
    assert row["question"] == "What is the Bank Rate?"
    assert row["kind"] == "one paragraph"
    assert row["found the answer?"] == "yes"
    assert row["first hit at"] == "1"


def test_question_rows_shorten_a_long_question():
    run = _run([QuestionResult("q001", ("a#0000",), ("a#0000",), True)])
    long_text = "What is the minimum proportion " * 10
    row = question_rows(run, [_question("q001", text=long_text)])[0]
    assert len(row["question"]) <= 78
    assert row["question"].endswith("…")
    # Cut on a word boundary, so the row does not end in half a word.
    kept = row["question"][:-1]
    assert long_text.startswith(kept)
    assert long_text[len(kept)] == " "


def test_question_rows_do_not_call_a_partial_hit_found():
    """q013 answers from one labelled paragraph while the second sits at rank 15,
    so a row saying only "found: yes" would be true and misleading."""
    retrieved = tuple(f"x#000{i}" for i in range(13)) + ("a#0013",) + ("y#0000",) * 6
    run = _run([QuestionResult("q013", retrieved, ("a#0000", "a#0013"), True)])
    row = question_rows(run, [_question("q013", relevant=("a#0000", "a#0013"))])[0]
    assert row["found the answer?"] == "partly"
    assert row["first hit at"] == "14"


def test_question_rows_call_both_paragraphs_found_when_both_are_in_reach():
    run = _run([QuestionResult("q014", ("a#0001", "b#0002"), ("a#0001", "b#0002"), True)])
    row = question_rows(run, [_question("q014", relevant=("a#0001", "b#0002"))])[0]
    assert row["found the answer?"] == "yes"


def test_question_rows_mark_a_question_retrieval_never_found():
    """q008 is the one answerable question this baseline retrieves at no depth."""
    run = _run([QuestionResult("q008", ("x#0000",), ("a#0000",), False)])
    row = question_rows(run, [_question("q008")])[0]
    assert row["found the answer?"] == "no"
    assert row["first hit at"] == "never"


def test_question_rows_cannot_score_an_unanswerable_question_as_a_miss():
    """There is no labelled chunk to find, so a miss would be the correct result
    being reported as a failure."""
    run = _run([QuestionResult("q019", ("x#0000",), (), False)])
    row = question_rows(
        run, [_question("q019", category="unanswerable", relevant=(), answerable=False)]
    )[0]
    assert row["kind"] == "not in this corpus"
    assert row["found the answer?"] == "n/a"
    assert row["first hit at"] == "-"


def test_answers_report_splits_the_summary_from_the_per_answer_detail():
    markdown = (
        "# Answer evaluation\n\n**16 of 18 answered correctly.**\n\n"
        "| qid |\n| --- |\n\n### q001 - single_clause\n\n**Verdict.** yes\n"
    )
    summary, detail = answers_report(markdown)
    assert summary.startswith("# Answer evaluation")
    assert "16 of 18" in summary
    assert "q001" not in summary
    assert detail.startswith("### q001")


def test_answers_report_handles_a_report_with_no_detail_yet():
    summary, detail = answers_report("# Answer evaluation\n\nnothing yet\n")
    assert "nothing yet" in summary
    assert detail == ""


def test_judge_client_is_none_when_no_key_is_configured():
    """The page must still render the evaluation tab without a key, so the button
    is hidden rather than the page broken."""
    assert judge_client(Settings(_env_file=None, llm_api_key="")) is None


# --- answering once per question ------------------------------------------


class _FakeSt:
    """Just enough of module `st` for the caching helper: session state and a
    spinner that does nothing."""

    def __init__(self) -> None:
        self.session_state: dict = {}

    def spinner(self, _message: str):
        from contextlib import nullcontext

        return nullcontext()


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict]) -> str:
        self.calls += 1
        return "The Bank Rate is 6.50 per cent [1]."


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0
        self.hit = Retrieved(
            chunk=Chunk("d#0000", "d", 0, "text zero", "text zero", {}),
            score=0.9,
            rank=1,
        )

    def retrieve(self, question: str, k: int) -> list[Retrieved]:
        self.calls += 1
        return [self.hit]


def test_a_question_is_answered_once_however_often_the_page_re_runs():
    """Streamlit re-runs the whole script on every click, so pressing the grade
    button would otherwise buy a second answer and grade a different one than the
    reader can see."""
    st, client, retriever = _FakeSt(), _FakeClient(), _FakeRetriever()
    first = _answer_entry(st, "same question", retriever, client)
    second = _answer_entry(st, "same question", retriever, client)
    assert first is second
    assert client.calls == 1
    assert retriever.calls == 1


def test_a_different_question_replaces_the_cached_answer():
    st, client, retriever = _FakeSt(), _FakeClient(), _FakeRetriever()
    _answer_entry(st, "first question", retriever, client)
    entry = _answer_entry(st, "second question", retriever, client)
    assert entry["question"] == "second question"
    assert client.calls == 2


def test_without_a_key_the_passages_are_kept_and_nothing_is_called():
    st, client, retriever = _FakeSt(), _FakeClient(), _FakeRetriever()
    entry = _answer_entry(st, "a question", retriever, None)
    assert entry["result"] is None
    assert entry["error"] is None
    assert entry["hits"] == [retriever.hit]
    assert client.calls == 0
