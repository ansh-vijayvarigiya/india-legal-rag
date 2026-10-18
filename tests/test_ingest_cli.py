from pathlib import Path

from legalrag.ingest import main

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def test_ingest_from_snapshots_writes_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    code = main(["--source", "rbi", "--snapshots", str(FIXTURES)])
    assert code == 0
    chunks = (tmp_path / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert chunks.count("\n") > 0
