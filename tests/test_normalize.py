import pytest

from legalrag.normalize import normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("bene\ufb01t under the scheme", "benefit under the scheme"),
        ("con\ufb02ict of laws", "conflict of laws"),
        ("the under-\nstanding of the Board", "the understanding of the Board"),
        ("tax   payable\therein", "tax payable herein"),
        ("Para one.\n\n\n\nPara two.", "Para one.\n\nPara two."),
        ("rate of GST is 18% \u2013 applicable", "rate of GST is 18% \u2013 applicable"),
    ],
)
def test_normalize_text_rules(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_preserves_devanagari():
    raw = "\u0930\u093f\u091c\u0930\u094d\u0935 \u092c\u0948\u0902\u0915"
    assert normalize_text(raw) == raw
