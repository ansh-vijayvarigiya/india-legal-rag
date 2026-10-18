from pathlib import Path

import pytest

from legalrag.sources.rbi import NotificationRef, ParseError, parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "rbi" / "notification_13696.html"

REF = NotificationRef(
    source_id="13696",
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13696&Mode=0",
    title="Review of Circulars issued under Foreign Exchange Management Act, 1999 (FEMA)",
    published="2026-09-08",
)


def test_parse_document_extracts_body_text():
    doc = parse_document(REF, FIXTURE.read_text(encoding="utf-8"))
    assert doc.doc_id == "rbi:13696"
    assert doc.source == "rbi"
    assert doc.source_url == REF.url
    assert doc.published == "2026-09-08"
    assert "shall" in doc.text or "circular" in doc.text.lower()
    assert len(doc.text) > 200
    assert doc.metadata["published"] == "2026-09-08"
    assert doc.metadata["circular"] == "RBI/2026-27/254"


def test_parse_document_keeps_non_ascii():
    doc = parse_document(REF, FIXTURE.read_text(encoding="utf-8"))
    assert "\u2019" in doc.text or "\u2013" in doc.text


def test_parse_document_rejects_unrecognised_page():
    ref = NotificationRef("99999", "https://www.rbi.org.in/x", "Broken", None)
    with pytest.raises(ParseError):
        parse_document(ref, "<html><body><p>Nothing useful here.</p></body></html>")
