from pathlib import Path

from legalrag.corpus import read_chunks
from legalrag.eval.__main__ import main
from legalrag.ingest import main as ingest_main

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def test_eval_cli_writes_results_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # The golden set can only reference real chunk ids, so ingest first.
    assert ingest_main(["--source", "rbi", "--snapshots", str(FIXTURES)]) == 0
    chunk_id = next(iter(read_chunks(tmp_path / "chunks" / "chunks.jsonl"))).chunk_id
    (tmp_path / "golden.yaml").write_text(
        "- qid: q001\n"
        "  question: What does this circular say about risk management?\n"
        f'  relevant_chunk_ids: ["{chunk_id}"]\n'
        "  category: single_clause\n"
        "  answerable: true\n"
        "- qid: q002\n"
        "  question: What is the GST rate on unicorn feed?\n"
        "  relevant_chunk_ids: []\n"
        "  category: unanswerable\n"
        "  answerable: false\n",
        encoding="utf-8",
    )

    assert main(["--ingest-fixtures", str(FIXTURES), "--golden", "golden.yaml"]) == 0
    assert "recall@5" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_eval_cli_reports_a_missing_golden_set(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert ingest_main(["--source", "rbi", "--snapshots", str(FIXTURES)]) == 0
    assert main(["--golden", "absent.yaml"]) == 1


def test_eval_cli_reports_an_empty_golden_set(tmp_path, monkeypatch):
    # An empty set scores 0.00 everywhere, which reads as a measurement.
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert ingest_main(["--source", "rbi", "--snapshots", str(FIXTURES)]) == 0
    (tmp_path / "golden.yaml").write_text("# no questions yet\n", encoding="utf-8")
    assert main(["--golden", "golden.yaml"]) == 1
