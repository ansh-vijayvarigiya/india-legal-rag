from pathlib import Path

import pytest

from legalrag.sources.rbi import NotificationRef, ParseError, parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "rbi" / "13739.html"

REF = NotificationRef(
    source_id="13739",
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13739&Mode=0",
    title="Risk Management and Inter-Bank Dealings",
    published="2026-10-12",
)


def test_parse_document_extracts_body_text():
    doc = parse_document(REF, FIXTURE.read_text(encoding="utf-8"))
    assert doc.doc_id == "rbi:13739"
    assert doc.source == "rbi"
    assert doc.source_url == REF.url
    assert doc.published == "2026-10-12"
    assert "shall" in doc.text
    assert len(doc.text) > 200
    assert doc.metadata["published"] == "2026-10-12"
    assert doc.metadata["circular"] == "RBI/2026-27/293"


def test_parse_document_rejects_unrecognised_page():
    ref = NotificationRef("99999", "https://www.rbi.org.in/x", "Broken", None)
    with pytest.raises(ParseError):
        parse_document(ref, "<html><body><p>Nothing useful here.</p></body></html>")
