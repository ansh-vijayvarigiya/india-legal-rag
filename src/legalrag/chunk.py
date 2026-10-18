"""The naive fixed-size baseline chunker.

It splits on blank lines and knows nothing about legal structure — no section,
rule, proviso or amendment boundaries. Structure-aware chunking is a later
milestone, and the ablation table is what justifies replacing this.
"""

from __future__ import annotations

from legalrag.models import Chunk, Document, make_chunk_id

_PARAGRAPH_SEPARATOR = "\n\n"


def _tail(text: str, chars: int) -> str:
    return text[-chars:] if chars > 0 else ""


def chunk_document(
    doc: Document, *, max_chars: int = 2000, overlap_chars: int = 200
) -> list[Chunk]:
    """Split a document into chunks of at most `max_chars`, overlapping by `overlap_chars`."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    # An overlap at or above the budget leaves no room for new text, so the loop
    # could never make progress. Half a budget always leaves room for something.
    overlap = min(overlap_chars, max_chars // 2)

    paragraphs = [p.strip() for p in doc.text.split(_PARAGRAPH_SEPARATOR)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []

    texts: list[str] = []
    buffer = ""
    pending = list(paragraphs)
    # True when `buffer` holds nothing but the overlap tail of a chunk already emitted.
    trailing_overlap_only = False

    while pending:
        paragraph = pending.pop(0)
        separator = _PARAGRAPH_SEPARATOR if buffer else ""
        room = max_chars - len(buffer) - len(separator)

        if room <= 0:
            texts.append(buffer)
            buffer = _tail(buffer, overlap)
            pending.insert(0, paragraph)
            trailing_overlap_only = True
            continue

        if len(paragraph) <= room:
            buffer = buffer + separator + paragraph
            trailing_overlap_only = False
            continue

        head, rest = paragraph[:room], paragraph[room:].strip()
        buffer = buffer + separator + head
        texts.append(buffer)
        buffer = _tail(buffer, overlap)
        trailing_overlap_only = True
        if rest:
            pending.insert(0, rest)

    if buffer and not trailing_overlap_only:
        texts.append(buffer)

    return [
        Chunk(
            chunk_id=make_chunk_id(doc.doc_id, ordinal),
            doc_id=doc.doc_id,
            ordinal=ordinal,
            text=text,
            embed_text=text,
            metadata=dict(doc.metadata),
        )
        for ordinal, text in enumerate(texts)
    ]
