from legalrag.answer import LLMError
from legalrag.inspector import example_questions, rate_limit_message, results_rows
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
