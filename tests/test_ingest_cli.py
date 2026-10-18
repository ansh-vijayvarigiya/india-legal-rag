from pathlib import Path

from legalrag.ingest import main

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def test_ingest_from_snapshots_writes_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    code = main(["--source", "rbi", "--snapshots", str(FIXTURES)])
    assert code == 0
    chunks = (tmp_path / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert chunks.count("\n") > 0


def test_ingest_fails_when_no_page_parses(tmp_path, monkeypatch):
    # Every page failing to parse is a regression, not a small partial corpus.
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "index.html").write_text(
        '<table><tr><td><a href="NotificationUser.aspx?Id=1">Notice One</a></td></tr></table>',
        encoding="utf-8",
    )
    (snapshots / "1.html").write_text(
        "<html><body><p>nothing here</p></body></html>", encoding="utf-8"
    )
    assert main(["--source", "rbi", "--snapshots", str(snapshots)]) == 1
