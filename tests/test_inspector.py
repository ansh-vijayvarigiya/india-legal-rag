from legalrag.inspector import results_rows
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
