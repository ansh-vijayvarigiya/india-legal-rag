"""The evaluation CLI. `python -m legalrag.eval`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legalrag.config import Settings
from legalrag.corpus import read_chunks
from legalrag.embed import Embedder
from legalrag.eval.report import render_report, write_report
from legalrag.eval.runner import DEFAULT_K_MAX, run_eval, validate_golden_against_corpus
from legalrag.golden import GoldenSetError, load_golden_set
from legalrag.index import ChunkIndex
from legalrag.ingest import main as ingest_main
from legalrag.retrieve import DenseRetriever

CONFIGURATIONS = ("dense-fixed",)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="legalrag.eval")
    parser.add_argument("--golden", type=Path, default=None, help="golden set YAML")
    parser.add_argument("--config", default=None, choices=CONFIGURATIONS)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--report", type=Path, default=Path("results.md"))
    parser.add_argument(
        "--ingest-fixtures",
        type=Path,
        default=None,
        help="ingest local HTML snapshots instead of the network",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings()

    if args.ingest_fixtures is not None:
        code = ingest_main(["--source", "rbi", "--snapshots", str(args.ingest_fixtures)])
        if code != 0:
            return code

    chunk_path = settings.data_dir / "chunks" / "chunks.jsonl"
    chunks = list(read_chunks(chunk_path))
    if not chunks:
        print(f"error: no chunks at {chunk_path}, ingest first", file=sys.stderr)
        return 1

    golden_path = args.golden or settings.data_dir / "golden" / "milestone1.yaml"
    try:
        questions = load_golden_set(golden_path)
    except FileNotFoundError:
        print(f"error: golden set not found at {golden_path}", file=sys.stderr)
        return 1
    except GoldenSetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # A label pointing at a chunk the corpus no longer has must fail loudly here,
    # where the corpus is known, rather than scoring zero.
    try:
        validate_golden_against_corpus(questions, [chunk.chunk_id for chunk in chunks])
    except GoldenSetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    embedder = Embedder(settings.embedding_model)
    collection = args.collection or settings.collection
    index = ChunkIndex(settings.qdrant_path, collection, embedder.dim)
    try:
        index.recreate()
        index.upsert(chunks, embedder.encode([chunk.embed_text for chunk in chunks]))
        retriever = DenseRetriever(index, embedder)
        configurations = [args.config] if args.config else list(CONFIGURATIONS)
        runs = [
            run_eval(retriever, questions, config=name, k_max=args.k_max)
            for name in configurations
        ]
    finally:
        index.close()

    markdown = render_report(runs)
    write_report(markdown, args.report)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
