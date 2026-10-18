import pytest

from legalrag.golden import GoldenSetError, load_golden_set

VALID = """
- qid: q001
  question: What is the time limit under Rule 36(4)?
  relevant_chunk_ids: ["cgst:rules#0007"]
  category: single_clause
  answerable: true
  gold_answer: Sixty days.
- qid: q002
  question: What is the GST rate on unicorn feed?
  relevant_chunk_ids: []
  category: unanswerable
  answerable: false
"""


def _write(tmp_path, body: str):
    path = tmp_path / "g.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_valid_golden_set(tmp_path):
    questions = load_golden_set(_write(tmp_path, VALID))
    assert [q.qid for q in questions] == ["q001", "q002"]
    assert questions[0].relevant_chunk_ids == ("cgst:rules#0007",)
    assert questions[1].gold_answer is None


def test_rejects_duplicate_qid(tmp_path):
    duplicate = (
        "- qid: q001\n"
        "  question: A second question reusing the id\n"
        '  relevant_chunk_ids: ["cgst:rules#0007"]\n'
        "  category: single_clause\n"
        "  answerable: true\n"
    )
    body = VALID + duplicate
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_set(_write(tmp_path, body))


def test_rejects_unknown_category(tmp_path):
    with pytest.raises(GoldenSetError, match="category"):
        load_golden_set(_write(tmp_path, VALID.replace("single_clause", "guessing")))


def test_rejects_missing_required_field(tmp_path):
    with pytest.raises(GoldenSetError, match="question"):
        load_golden_set(_write(tmp_path, "- qid: q003\n  category: single_clause\n"))


def test_rejects_answerable_question_without_relevant_ids(tmp_path):
    body = VALID.replace('relevant_chunk_ids: ["cgst:rules#0007"]', "relevant_chunk_ids: []")
    with pytest.raises(GoldenSetError, match="relevant_chunk_ids"):
        load_golden_set(_write(tmp_path, body))


def test_comment_only_file_loads_as_empty(tmp_path):
    assert load_golden_set(_write(tmp_path, "# nothing here yet\n")) == []
