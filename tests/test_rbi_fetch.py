from pathlib import Path

from legalrag.sources.rbi import parse_index

FIXTURE = Path(__file__).parent / "fixtures" / "rbi" / "index.html"


def test_parse_index_extracts_notification_refs():
    refs = parse_index(FIXTURE.read_text(encoding="utf-8"))
    assert len(refs) > 0
    assert all(ref.url.startswith("https://www.rbi.org.in/") for ref in refs)
    assert all(ref.source_id for ref in refs)
    assert all(ref.title for ref in refs)
    assert len({ref.source_id for ref in refs}) == len(refs)
