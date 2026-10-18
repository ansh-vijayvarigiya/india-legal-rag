import re

# Order matters: ligatures and hyphenated line breaks are repaired before
# whitespace is collapsed, or the break is lost and the word stays split.
_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl"}
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_AND_TABS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(raw: str) -> str:
    """Normalise gazette and web text while preserving non-ASCII content."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = _SPACES_AND_TABS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()
