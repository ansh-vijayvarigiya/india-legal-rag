"""Fetch, parse and chunk RBI notification pages into the corpus store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legalrag.chunk import chunk_document
from legalrag.config import Settings
from legalrag.corpus import write_chunks, write_documents
from legalrag.models import Chunk, Document
from legalrag.sources import rbi

CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="legalrag.ingest")
    parser.add_argument("--source", default="rbi", choices=["rbi"])
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="read local HTML snapshots instead of the network",
    )
    parser.add_argument("--limit", type=int, default=None, help="at most N notifications")
    return parser.parse_args(argv)


def _fetch(raw_dir: Path, limit: int | None) -> None:
    with rbi.make_client() as client:
        index_path = rbi.fetch_index(client, raw_dir)
        refs = rbi.parse_index(index_path.read_text(encoding="utf-8"))
        for ref in refs[:limit] if limit else refs:
            rbi.fetch_document(client, ref, raw_dir)


def _indexed_refs(
    index_path: Path, snapshot_dir: Path, limit: int | None
) -> list[rbi.NotificationRef]:
    """Index entries that have a snapshot on disk, so a partial corpus is fine."""
    refs = [
        ref
        for ref in rbi.parse_index(index_path.read_text(encoding="utf-8"))
        if (snapshot_dir / f"{ref.source_id}.html").exists()
    ]
    return refs[:limit] if limit else refs


def _parse_documents(
    refs: list[rbi.NotificationRef], snapshot_dir: Path
) -> tuple[list[Document], int]:
    documents: list[Document] = []
    failures = 0
    for ref in refs:
        html = (snapshot_dir / f"{ref.source_id}.html").read_text(encoding="utf-8")
        try:
            documents.append(rbi.parse_document(ref, html))
        except rbi.ParseError as error:
            failures += 1
            print(f"skipped {ref.source_id}: {error}", file=sys.stderr)
    return documents, failures


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # A CLI run is a single process, so read the environment fresh rather than
    # through the cached accessor the library and UI use.
    settings = Settings()
    snapshot_dir = args.snapshots or settings.data_dir / "raw" / args.source

    if args.snapshots is None:
        _fetch(settings.data_dir / "raw" / args.source, args.limit)

    index_path = snapshot_dir / "index.html"
    if not index_path.exists():
        print(f"error: no index snapshot at {index_path}", file=sys.stderr)
        return 1

    refs = _indexed_refs(index_path, snapshot_dir, args.limit)
    documents, failures = _parse_documents(refs, snapshot_dir)
    chunks: list[Chunk] = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS
        )
    ]

    write_documents(settings.data_dir / "documents" / "documents.jsonl", documents)
    write_chunks(settings.data_dir / "chunks" / "chunks.jsonl", chunks)

    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    print(f"parse failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
