from legalrag.models import Chunk, Document, Retrieved, make_chunk_id


def test_make_chunk_id_zero_pads_ordinal():
    assert make_chunk_id("rbi:13121", 7) == "rbi:13121#0007"


def test_chunk_json_round_trip_preserves_fields():
    chunk = Chunk(
        chunk_id="rbi:13121#0000",
        doc_id="rbi:13121",
        ordinal=0,
        text="Payment of tax.",
        embed_text="Payment of tax.",
        metadata={"circular": "RBI/2026-27/254"},
    )
    assert Chunk.from_json(chunk.to_json()) == chunk


def test_document_json_round_trip_preserves_fields():
    doc = Document(
        doc_id="rbi:13121",
        source="rbi",
        source_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13121&Mode=0",
        title="Review of Circulars issued under FEMA",
        published="2026-09-08",
        text="In pursuance of the Reserve Bank's ongoing initiative",
        metadata={"circular": "RBI/2026-27/254"},
    )
    assert Document.from_json(doc.to_json()) == doc


def test_retrieved_holds_chunk_score_and_rank():
    chunk = Chunk("d#0000", "d", 0, "text", "text", {})
    retrieved = Retrieved(chunk=chunk, score=0.87, rank=1)
    assert retrieved.rank == 1
