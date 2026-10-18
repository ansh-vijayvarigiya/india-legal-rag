"""JSON Lines storage for the parsed corpus."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from legalrag.models import Chunk, Document


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write one JSON object per line, UTF-8, and return the record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield each record. A missing file yields nothing rather than raising."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_documents(path: Path, documents: Iterable[Document]) -> int:
    return write_jsonl(path, (document.to_json() for document in documents))


def read_documents(path: Path) -> Iterator[Document]:
    return (Document.from_json(record) for record in read_jsonl(path))


def write_chunks(path: Path, chunks: Iterable[Chunk]) -> int:
    return write_jsonl(path, (chunk.to_json() for chunk in chunks))


def read_chunks(path: Path) -> Iterator[Chunk]:
    return (Chunk.from_json(record) for record in read_jsonl(path))
