"""Guards on the committed artifacts: the golden set and the corpus it labels.

These are offline and need no model. They exist because the baseline numbers in
`results.md` and `README.md` are claims about these files, and nothing else
checks them: a mislabelled category, a duplicated qid, or a label left behind by
a re-chunk would otherwise surface only to a human running the evaluation.
"""

from collections import Counter
from pathlib import Path

from legalrag.corpus import read_chunks
from legalrag.eval.runner import validate_golden_against_corpus
from legalrag.golden import load_golden_set
from legalrag.ingest import main as ingest_main

REPO = Path(__file__).parent.parent
GOLDEN = REPO / "data" / "golden" / "milestone1.yaml"
SNAPSHOTS = REPO / "data" / "raw" / "rbi"

# The mix the design asks for, and the corpus size the README reports.
EXPECTED_MIX = {
    "single_clause": 10,
    "multi_clause": 4,
    "cross_document": 2,
    "temporal": 2,
    "unanswerable": 2,
}
EXPECTED_DOCUMENTS = 25
EXPECTED_CHUNKS = 154


def test_committed_golden_set_has_the_expected_mix():
    questions = load_golden_set(GOLDEN)
    assert len(questions) == 20
    assert len({question.qid for question in questions}) == 20
    assert Counter(question.category for question in questions) == Counter(EXPECTED_MIX)
    for question in questions:
        assert question.answerable == bool(question.relevant_chunk_ids), question.qid


def test_committed_labels_exist_in_the_committed_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    assert ingest_main(["--source", "rbi", "--snapshots", str(SNAPSHOTS)]) == 0

    chunk_ids = [
        chunk.chunk_id for chunk in read_chunks(tmp_path / "chunks" / "chunks.jsonl")
    ]
    # The counts the README states. A change here renumbers chunks and the
    # golden set has to be re-verified, so failing loudly is the point.
    assert len(chunk_ids) == EXPECTED_CHUNKS
    documents = (tmp_path / "documents" / "documents.jsonl").read_text(
        encoding="utf-8"
    )
    assert documents.count("\n") == EXPECTED_DOCUMENTS

    # Raises GoldenSetError if any label points at a chunk the corpus lacks.
    validate_golden_against_corpus(load_golden_set(GOLDEN), chunk_ids)
