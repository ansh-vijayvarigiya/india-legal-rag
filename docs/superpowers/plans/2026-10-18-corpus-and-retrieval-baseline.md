# Corpus and Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest RBI regulatory documents and produce a measured dense-retrieval baseline (configuration 0 from the spec) against a hand-verified 20-question golden set.

**Architecture:** A linear offline pipeline — fetch live RBI pages into raw HTML snapshots, parse them into `Document` records, split into fixed-size `Chunk`s, embed locally with `bge-small-en-v1.5`, and index into Qdrant in local on-disk mode. A separate evaluation path loads the golden set, runs a retriever over it, computes retrieval metrics with no language model involved, and renders `results.md`. Every stage reads and writes JSONL or HTML artifacts so any stage can be re-run alone.

**Tech Stack:** Python 3.11, `pydantic-settings`, `httpx`, `selectolax`, `numpy`, `qdrant-client` (local mode), `sentence-transformers`, `PyYAML`, `pytest`, `make`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-18-indian-regulatory-rag-design.md`

## Scope of This Plan

This plan covers spec milestone 1, split in two because it is too large for one plan. This half ends with measured retrieval numbers and costs nothing to run.

**Deferred to the next plan:** answer generation with citations, refusal behaviour, the LLM judge and its calibration, the FastAPI service, and trace logging.

**Deferred to later plans:** the India Code and CBIC sources (this plan ingests RBI only, because its pages carry full text as HTML and need no PDF parsing), structure-aware chunking (spec section 5), the metadata and temporal model (section 6), hybrid retrieval with RRF, cross-encoder reranking, and query transformation (section 7). Each is a row in the ablation table that this plan's harness exists to measure.

Nothing here is optional filler. The golden set, the metrics and the report are what every later change is measured against, which is why they come before the retrieval work they judge.

## Global Constraints

- Python 3.11 (the interpreter on `PATH`). Pin in `pyproject.toml` with `requires-python = ">=3.11,<3.12"`.
- Windows, PowerShell. No Docker on this machine — do not introduce Postgres, pgvector, or any service requiring a container.
- Qdrant runs in local on-disk mode only: `QdrantClient(path=...)`.
- Embeddings run locally via `sentence-transformers`. No embedding API, no API key anywhere in this plan.
- Package name is `legalrag`; repository is `india-legal-rag`; all source lives under `src/legalrag/`.
- Retrieval metric key set is `k = 5, 10, 20`. Evaluation retrieves `k_max = 20`.
- Golden set is committed at `data/golden/milestone1.yaml` and is version-controlled.
- Generated report path is `results.md` at the repository root.
- No language model is called anywhere in this plan. Answer generation, citation validity, refusal and the LLM judge belong to the following plan.

## Review Focus

Five input classes the spec implies but whose tests no task would otherwise write. Each is pinned to a test in the task that owns the code.

1. **Empty or zero-result retrieval.** A retriever returning nothing (empty corpus, no lexical overlap) must not produce `NaN`, must not divide by zero in MRR or nDCG, and must not crash the report. Expected: `0.0` for every metric. Pinned in Task 12.
2. **A golden-set `relevant_chunk_ids` entry that no longer exists in the corpus.** Re-ingestion changes chunk ids. Expected: a hard failure naming the missing id, not a silent score of zero. Pinned in Task 13.
3. **Questions with more than one relevant chunk.** Cross-document and multi-clause questions list several ids. Expected: recall and nDCG use the full relevant set, and precision is computed against `k`, not against `len(relevant)`. Pinned in Task 12.
4. **Non-ASCII text.** RBI pages contain Hindi, Devanagari punctuation, and typographic quotes. Expected: normalisation preserves them rather than dropping or mojibake-ing them. Pinned in Task 3.
5. **An RBI page whose structure does not match the expected shape** (site redesign, error page, truncated response). Expected: the parser raises a named error instead of emitting an empty `Document` that silently pollutes the corpus. Pinned in Task 5.

---

### Task 1: Project scaffold and settings

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/legalrag/__init__.py`
- Create: `src/legalrag/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (fields `data_dir: Path`, `qdrant_path: Path`, `embedding_model: str`, `collection: str`), and `get_settings() -> Settings`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from legalrag.config import Settings, get_settings


def test_settings_have_expected_defaults():
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.qdrant_path == Path("data/qdrant")


def test_env_var_overrides_data_dir(monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", "D:/tmp/legalrag")
    get_settings.cache_clear()
    assert get_settings().data_dir == Path("D:/tmp/legalrag")
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag'`

- [ ] **Step 3: Implement the scaffold and `Settings`**

Create `pyproject.toml` with `[project] name = "legalrag"`, `requires-python = ">=3.11,<3.12"`, dependencies `pydantic-settings`, `httpx`, `selectolax`, `numpy`, `qdrant-client`, `sentence-transformers`, `PyYAML`, and `pytest` under `[project.optional-dependencies] dev`. Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `pythonpath = ["src"]`. Add a `slow` marker used by Tasks 8 and 9.

`Settings` subclasses `BaseSettings` with `model_config = SettingsConfigDict(env_prefix="LEGALRAG_")` and the four fields above. `get_settings` is decorated with `functools.lru_cache`. `qdrant_path` and `collection` have defaults; do not require any environment variable to be set.

`.gitignore` ignores `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/raw/`, `data/parsed/`, `data/qdrant/`, `data/chunks/`, `data/cache/`. It must **not** ignore `data/golden/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/legalrag tests/test_config.py
git commit -m "feat: add project scaffold and settings"
```

---

### Task 2: Core data models

**Files:**
- Create: `src/legalrag/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Document`, `Chunk`, `Retrieved`, and `make_chunk_id(doc_id: str, ordinal: int) -> str`.

- [ ] **Step 1: Write the failing test**

```python
from legalrag.models import Chunk, Document, Retrieved, make_chunk_id


def test_make_chunk_id_zero_pads_ordinal():
    assert make_chunk_id("rbi:13121", 7) == "rbi:13121#0007"


def test_chunk_json_round_trip_preserves_fields():
    chunk = Chunk(
        chunk_id="rbi:13121#0000",
        doc_id="rbi:13121",
        ordinal=0,
        text="Payment of tax.",
        embed_text="Payment of tax.",
        metadata={"circular": "RBI/2026-27/254"},
    )
    assert Chunk.from_json(chunk.to_json()) == chunk


def test_document_json_round_trip_preserves_fields():
    doc = Document(
        doc_id="rbi:13121",
        source="rbi",
        source_url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13121&Mode=0",
        title="Review of Circulars issued under FEMA",
        published="2026-09-08",
        text="In pursuance of the Reserve Bank's ongoing initiative",
        metadata={"circular": "RBI/2026-27/254"},
    )
    assert Document.from_json(doc.to_json()) == doc


def test_retrieved_holds_chunk_score_and_rank():
    chunk = Chunk("d#0000", "d", 0, "text", "text", {})
    retrieved = Retrieved(chunk=chunk, score=0.87, rank=1)
    assert retrieved.rank == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.models'`

- [ ] **Step 3: Implement the models**

All four are `@dataclass(frozen=True, slots=True)`. `Document` fields: `doc_id`, `source`, `source_url`, `title`, `published: str | None` (ISO 8601 date), `text`, `metadata: dict[str, str]`. `Chunk` fields: `chunk_id`, `doc_id`, `ordinal: int`, `text`, `embed_text`, `metadata: dict[str, str]`. `Retrieved` fields: `chunk: Chunk`, `score: float`, `rank: int`.

`to_json()` returns a plain `dict`; `from_json(obj)` is a classmethod that rebuilds the dataclass. `make_chunk_id` returns `f"{doc_id}#{ordinal:04d}"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/models.py tests/test_models.py
git commit -m "feat: add document, chunk and retrieved models"
```

---

### Task 3: Text normalisation

**Files:**
- Create: `src/legalrag/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_text(raw: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from legalrag.normalize import normalize_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("e\ufb01ect of the said rule", "effect of the said rule"),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.normalize'`

- [ ] **Step 3: Implement `normalize_text`**

In order: replace `\r\n` and `\r` with `\n`; expand the ligatures `\ufb01` and `\ufb02` to `fi` and `fl`; join hyphenated line breaks (`(\w)-\n(\w)` to `\1\2`); strip zero-width and control characters except `\n`; collapse runs of spaces and tabs within a line to a single space; collapse three or more newlines to exactly two; strip trailing spaces per line and leading/trailing whitespace overall. Leave every non-ASCII printable character untouched — Devanagari, en and em dashes and typographic quotes all pass through.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/normalize.py tests/test_normalize.py
git commit -m "feat: add legal text normalisation"
```

---

### Task 4: RBI fetch into raw snapshots

**Files:**
- Create: `src/legalrag/sources/__init__.py`
- Create: `src/legalrag/sources/rbi.py`
- Create: `tests/fixtures/rbi/index.html`
- Create: `tests/fixtures/rbi/notification_13121.html`
- Test: `tests/test_rbi_fetch.py`

**Interfaces:**
- Produces: `NotificationRef` (frozen dataclass: `source_id: str`, `url: str`, `title: str`, `published: str | None`), `ParseError(Exception)`, `parse_index(html: str) -> list[NotificationRef]`, `fetch_index(client: httpx.Client, snapshot_dir: Path) -> Path`, `fetch_document(client: httpx.Client, ref: NotificationRef, snapshot_dir: Path) -> Path`.

- [ ] **Step 1: Capture real fixtures**

Using `httpx`, fetch `https://www.rbi.org.in/Scripts/NotificationUser.aspx` and save the body as `tests/fixtures/rbi/index.html`. Fetch `https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13121&Mode=0` and save as `tests/fixtures/rbi/notification_13121.html`. Do this once, by hand, and commit both. Every test below runs offline against these files.

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_rbi_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.sources.rbi'`

- [ ] **Step 4: Implement fetching and index parsing**

`parse_index` walks the anchor tags matching `NotificationUser.aspx?Id=(\d+)`, deriving `source_id` from the numeric id and the absolute `url` from the href. Titles are the anchor text, whitespace-normalised. `published` is parsed from the surrounding row's date cell as an ISO 8601 date and is `None` when no date is present.

`fetch_index` and `fetch_document` use `httpx` with a descriptive `User-Agent` header, a 30-second timeout, `time.sleep(1)` between requests, and `raise_for_status()`. Each writes the response body to `snapshot_dir` and returns the written path. `fetch_document` writes to `{ref.source_id}.html`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_rbi_fetch.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add src/legalrag/sources tests/fixtures tests/test_rbi_fetch.py
git commit -m "feat: fetch and index RBI notification pages"
```

---

### Task 5: RBI HTML to Document

**Files:**
- Modify: `src/legalrag/sources/rbi.py`
- Test: `tests/test_rbi_parse.py`

**Interfaces:**
- Consumes: `NotificationRef`, `ParseError` from Task 4; `Document` from Task 2; `normalize_text` from Task 3.
- Produces: `parse_document(ref: NotificationRef, html: str) -> Document`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from legalrag.sources.rbi import NotificationRef, ParseError, parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "rbi" / "notification_13121.html"

REF = NotificationRef(
    source_id="13121",
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13121&Mode=0",
    title="Review of Circulars issued under FEMA",
    published="2026-09-08",
)


def test_parse_document_extracts_body_text():
    doc = parse_document(REF, FIXTURE.read_text(encoding="utf-8"))
    assert doc.doc_id == "rbi:13121"
    assert doc.source == "rbi"
    assert doc.source_url == REF.url
    assert doc.published == "2026-09-08"
    assert "shall" in doc.text or "circular" in doc.text.lower()
    assert len(doc.text) > 200
    assert doc.metadata["published"] == "2026-09-08"
    assert not doc.metadata.get("circular", "").startswith("")


def test_parse_document_rejects_unrecognised_page():
    ref = NotificationRef("99999", "https://www.rbi.org.in/x", "Broken", None)
    with pytest.raises(ParseError):
        parse_document(ref, "<html><body><p>Nothing useful here.</p></body></html>")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rbi_parse.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_document'`

- [ ] **Step 3: Implement `parse_document`**

Select the notification body container and extract its text; extract the circular number by matching the `RBI/yyyy-yy/NNN` pattern. Run everything through `normalize_text`. Set `doc_id` to `f"rbi:{ref.source_id}"` and `metadata` to `{"published": ..., "circular": ...}` using empty strings when a value is unknown.

Raise `ParseError` when the extracted text is shorter than 200 characters or when the body container is absent. This is Review Focus #5: a malformed or error page must fail loudly rather than emit an empty document.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rbi_parse.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/sources/rbi.py tests/test_rbi_parse.py
git commit -m "feat: parse RBI notification pages into documents"
```

---

### Task 6: Fixed-size chunker

**Files:**
- Create: `src/legalrag/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `Document`, `Chunk`, `make_chunk_id` from Task 2.
- Produces: `chunk_document(doc: Document, *, max_chars: int = 2000, overlap_chars: int = 200) -> list[Chunk]`.

This is the deliberate naive baseline for spec configuration 0. It splits on paragraph boundaries and knows nothing about legal structure; structure-aware chunking is Task 1 of the next plan, and the ablation table is what justifies replacing this.

- [ ] **Step 1: Write the failing test**

```python
from legalrag.chunk import chunk_document
from legalrag.models import Document, make_chunk_id


def _doc(text: str) -> Document:
    return Document("rbi:1", "rbi", "https://example.invalid/1", "T", None, text, {})


def test_chunks_are_contiguous_and_within_budget():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(6))
    chunks = chunk_document(_doc(text), max_chars=1000, overlap_chars=100)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [make_chunk_id("rbi:1", i) for i in range(len(chunks))]
    assert all(len(c.text) <= 1000 for c in chunks)


def test_overlap_carries_tail_text_into_the_next_chunk():
    text = "\n\n".join("word " * 100 for _ in range(6))
    chunks = chunk_document(_doc(text), max_chars=600, overlap_chars=200)
    assert len(chunks) > 1
    assert chunks[0].text[-100:] in chunks[1].text


def test_embed_text_equals_text_at_this_stage():
    chunks = chunk_document(_doc("A single short paragraph."))
    assert chunks[0].embed_text == chunks[0].text


def test_blank_document_produces_no_chunks():
    assert chunk_document(_doc("")) == []
    assert chunk_document(_doc("   \n\n  ")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.chunk'`

- [ ] **Step 3: Implement `chunk_document`**

Split `doc.text` on blank lines into paragraphs, strip empty ones, and accumulate paragraphs into a buffer. Flush the buffer when adding the next paragraph would exceed `max_chars`, then seed the new buffer with the last `overlap_chars` characters of the flushed text. A single paragraph longer than `max_chars` is hard-split. Return `[]` when the document has no paragraphs with content. Build each `Chunk` with `doc_id=doc.doc_id`, sequential `ordinal` from 0, `text` and `embed_text` identical, and `metadata` copied from the document.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chunk.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/chunk.py tests/test_chunk.py
git commit -m "feat: add fixed-size baseline chunker"
```

---

### Task 7: Corpus store and ingest CLI

**Files:**
- Create: `src/legalrag/corpus.py`
- Create: `src/legalrag/ingest.py`
- Test: `tests/test_corpus.py`
- Test: `tests/test_ingest_cli.py`

**Interfaces:**
- Consumes: `Document`, `Chunk` from Task 2; `chunk_document` from Task 6; `parse_document`, `parse_index` from Tasks 4 and 5; `Settings` from Task 1.
- Produces: `write_jsonl(path, records)`, `read_jsonl(path)`, `write_documents(path, docs)`, `read_documents(path)`, `write_chunks(path, chunks)`, `read_chunks(path)`, and `main(argv: list[str] | None = None) -> int` in `ingest.py`.

- [ ] **Step 1: Write the failing test**

```python
from legalrag.corpus import read_chunks, write_chunks
from legalrag.models import Chunk


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunks = [
        Chunk("rbi:1#0000", "rbi:1", 0, "alpha", "alpha", {"a": "b"}),
        Chunk("rbi:1#0001", "rbi:1", 1, "beta", "beta", {}),
    ]
    assert write_chunks(path, chunks) == 2
    assert list(read_chunks(path)) == chunks


def test_read_chunks_missing_file_returns_empty(tmp_path):
    assert list(read_chunks(tmp_path / "absent.jsonl")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.corpus'`

- [ ] **Step 3: Implement the corpus store**

One JSON object per line, UTF-8, `ensure_ascii=False`. `write_jsonl` returns the record count. `read_jsonl` is a generator yielding dicts. The typed helpers map to and from `to_json`/`from_json`. A missing path yields nothing rather than raising.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing CLI test**

```python
from pathlib import Path

from legalrag.ingest import main

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def test_ingest_from_snapshots_writes_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    code = main(["--source", "rbi", "--snapshots", str(FIXTURES)])
    assert code == 0
    chunks = (tmp_path / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert chunks.count("\n") > 0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'main'`

- [ ] **Step 7: Implement the ingest CLI**

`main` accepts `--source rbi`, `--snapshots PATH` (read local HTML instead of the network — this is what makes the test offline), and `--limit N`. With no `--snapshots` it calls `fetch_index` then `fetch_document` into `{data_dir}/raw/{source}/`. It then parses every snapshot into a `Document`, skipping and counting any that raise `ParseError`, chunks each document, and writes `{data_dir}/documents/documents.jsonl` and `{data_dir}/chunks/chunks.jsonl`. It prints the document count, chunk count and parse-failure count and returns 0.

Add an `if __name__ == "__main__": raise SystemExit(main())` guard so `python -m legalrag.ingest` works.

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 9: Commit**

```bash
git add src/legalrag/corpus.py src/legalrag/ingest.py tests/test_corpus.py tests/test_ingest_cli.py
git commit -m "feat: add corpus store and ingest CLI"
```

---

### Task 8: Local embeddings

**Files:**
- Create: `src/legalrag/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: `Embedder` with `__init__(model_name: str, *, device: str | None = None, batch_size: int = 32)`, property `dim: int`, and `encode(texts: Sequence[str]) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from legalrag.embed import Embedder

pytestmark = pytest.mark.slow


def test_encode_returns_normalised_vectors():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    vectors = embedder.encode(["payment of tax", "input tax credit"])
    assert vectors.shape == (2, embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_is_deterministic():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    first = embedder.encode(["same text"])
    second = embedder.encode(["same text"])
    assert np.allclose(first, second)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embed.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.embed'`

- [ ] **Step 3: Implement `Embedder`**

Wrap `sentence_transformers.SentenceTransformer`, loading once in `__init__`. `dim` comes from `get_sentence_embedding_dimension()`. `encode` calls `model.encode` with `normalize_embeddings=True`, `batch_size`, and `convert_to_numpy=True`, and returns `np.float32`. When `device` is `None`, prefer `"cuda"` and fall back to `"cpu"` if CUDA is unavailable — on Blackwell this needs a recent PyTorch build with CUDA 12.8 wheels, and CPU is an acceptable fallback for this corpus size. Encode an empty sequence as a `(0, dim)` array without calling the model.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embed.py -v -m slow`
Expected: PASS (2 tests). The first run downloads the model.

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/embed.py tests/test_embed.py
git commit -m "feat: add local bge embedding wrapper"
```

---

### Task 9: Qdrant chunk index

**Files:**
- Create: `src/legalrag/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `Chunk`, `Retrieved` from Task 2; `Settings` from Task 1.
- Produces: `ChunkIndex` with `__init__(path: Path, collection: str, dim: int)`, `recreate() -> None`, `upsert(chunks: Sequence[Chunk], vectors: np.ndarray) -> None`, `search(vector: np.ndarray, k: int) -> list[Retrieved]`, `get(chunk_id: str) -> Chunk | None`, `count() -> int`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from legalrag.index import ChunkIndex
from legalrag.models import Chunk

DIM = 4


def _chunk(i: int) -> Chunk:
    return Chunk(f"d#{i:04d}", "d", i, f"text {i}", f"text {i}", {})


def test_upsert_and_search_returns_ranked_chunks(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    chunks = [_chunk(0), _chunk(1)]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    index.upsert(chunks, vectors)
    assert index.count() == 2

    hits = index.search(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "d#0000"
    assert hits[0].score > hits[1].score


def test_get_round_trips_payload(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    index.upsert([_chunk(3)], np.array([[0, 0, 1, 0]], dtype=np.float32))
    assert index.get("d#0003") == _chunk(3)
    assert index.get("d#9999") is None


def test_recreate_empties_the_collection(tmp_path):
    index = ChunkIndex(tmp_path / "qdrant", "test", DIM)
    index.recreate()
    index.upsert([_chunk(0)], np.array([[1, 0, 0, 0]], dtype=np.float32))
    index.recreate()
    assert index.count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.index'`

- [ ] **Step 3: Implement `ChunkIndex`**

`QdrantClient(path=str(path))` — local on-disk mode, because Docker is unavailable. `recreate` deletes the collection if present and creates it with `VectorParams(size=dim, distance=Distance.COSINE)`. `upsert` writes point ids as the `chunk_id` string, vectors as float lists, and payload as the chunk's `to_json()` dict; use `wait=True`. `search` calls `query_points` with `limit=k` and maps results to `Retrieved` with `rank` starting at 1. `get` uses `retrieve` and returns the payload rebuilt through `Chunk.from_json`, or `None`. `count` returns `get_collection(...).points_count or 0`.

Keep every Qdrant call inside this module and expose only the methods above, so retrieval code can move to a Qdrant server later without change.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_index.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/index.py tests/test_index.py
git commit -m "feat: add local qdrant chunk index"
```

---

### Task 10: Dense retriever (configuration 0)

**Files:**
- Create: `src/legalrag/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `ChunkIndex` from Task 9; `Embedder` from Task 8; `Retrieved` from Task 2.
- Produces: a `Retriever` protocol with `retrieve(query: str, k: int = 10) -> list[Retrieved]`, and `DenseRetriever(index: ChunkIndex, embedder: Embedder)` implementing it.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from legalrag.index import ChunkIndex
from legalrag.models import Chunk
from legalrag.retrieve import DenseRetriever

VECTORS = {"alpha": [1, 0, 0], "beta": [0, 1, 0], "query about alpha": [1, 0, 0]}


class FakeEmbedder:
    dim = 3

    def encode(self, texts):
        return np.array([VECTORS[t] for t in texts], dtype=np.float32)


def _retriever(tmp_path) -> DenseRetriever:
    index = ChunkIndex(tmp_path / "qdrant", "test", 3)
    index.recreate()
    chunks = [Chunk("d#0000", "d", 0, "alpha", "alpha", {}), Chunk("d#0001", "d", 1, "beta", "beta", {})]
    index.upsert(chunks, np.array([VECTORS["alpha"], VECTORS["beta"]], dtype=np.float32))
    return DenseRetriever(index, FakeEmbedder())


def test_retrieve_returns_top_k_in_rank_order(tmp_path):
    hits = _retriever(tmp_path).retrieve("query about alpha", k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "d#0000"


def test_retrieve_respects_k(tmp_path):
    assert len(_retriever(tmp_path).retrieve("query about alpha", k=1)) == 1


def test_retrieve_on_empty_index_returns_empty(tmp_path):
    index = ChunkIndex(tmp_path / "empty", "test", 3)
    index.recreate()
    assert DenseRetriever(index, FakeEmbedder()).retrieve("alpha", k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.retrieve'`

- [ ] **Step 3: Implement `Retriever` and `DenseRetriever`**

`Retriever` is a `typing.Protocol` with the single `retrieve` method. `DenseRetriever.retrieve` encodes the query as a one-element sequence, takes the first row, and calls `index.search`. It returns `[]` when the index is empty. This class is the only retrieval configuration that exists at this milestone; Tasks 11 to 13 are written against the protocol so later configurations drop in without touching the evaluation code.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/retrieve.py tests/test_retrieve.py
git commit -m "feat: add dense retriever baseline"
```

---

### Task 11: Golden set schema and loader

**Files:**
- Create: `src/legalrag/golden.py`
- Create: `data/golden/milestone1.yaml`
- Test: `tests/test_golden.py`

**Interfaces:**
- Produces: `GoldenQuestion` (frozen dataclass: `qid: str`, `question: str`, `relevant_chunk_ids: tuple[str, ...]`, `category: str`, `answerable: bool`, `gold_answer: str | None`), `GoldenSetError(ValueError)`, `load_golden_set(path: Path) -> list[GoldenQuestion]`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from legalrag.golden import GoldenSetError, load_golden_set

VALID = """
- qid: q001
  question: What is the time limit under Rule 36(4)?
  relevant_chunk_ids: ["cgst:rules#0007"]
  category: single_clause
  answerable: true
  gold_answer: Sixty days.
- qid: q002
  question: What is the GST rate on unicorn feed?
  relevant_chunk_ids: []
  category: unanswerable
  answerable: false
"""


def _write(tmp_path, body: str):
    path = tmp_path / "g.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_valid_golden_set(tmp_path):
    questions = load_golden_set(_write(tmp_path, VALID))
    assert [q.qid for q in questions] == ["q001", "q002"]
    assert questions[0].relevant_chunk_ids == ("cgst:rules#0007",)
    assert questions[1].gold_answer is None


def test_rejects_duplicate_qid(tmp_path):
    duplicate = (
        "- qid: q001\n"
        "  question: A second question reusing the id\n"
        '  relevant_chunk_ids: ["cgst:rules#0007"]\n'
        "  category: single_clause\n"
        "  answerable: true\n"
    )
    body = VALID + duplicate
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_set(_write(tmp_path, body))


def test_rejects_unknown_category(tmp_path):
    with pytest.raises(GoldenSetError, match="category"):
        load_golden_set(_write(tmp_path, VALID.replace("single_clause", "guessing")))


def test_rejects_missing_required_field(tmp_path):
    with pytest.raises(GoldenSetError, match="question"):
        load_golden_set(_write(tmp_path, "- qid: q003\n  category: single_clause\n"))


def test_rejects_answerable_question_without_relevant_ids(tmp_path):
    body = VALID.replace('relevant_chunk_ids: ["cgst:rules#0007"]', "relevant_chunk_ids: []")
    with pytest.raises(GoldenSetError, match="relevant_chunk_ids"):
        load_golden_set(_write(tmp_path, body))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.golden'`

- [ ] **Step 3: Implement the schema and loader**

Allowed categories are exactly `single_clause`, `multi_clause`, `cross_document`, `temporal`, `unanswerable`. Validation raises `GoldenSetError` with a message naming the offending `qid` and field for: duplicate `qid`; missing or empty `question`; unknown `category`; `answerable` true with an empty `relevant_chunk_ids`; `category: unanswerable` with a non-empty `relevant_chunk_ids`; and a non-list `relevant_chunk_ids`. `relevant_chunk_ids` loads as a `tuple`. The file is a YAML list of mappings.

Also create `data/golden/milestone1.yaml` containing just a YAML comment describing the schema. It is filled in by Task 15. The loader must accept an empty or comment-only file and return `[]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_golden.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/golden.py data/golden/milestone1.yaml tests/test_golden.py
git commit -m "feat: add golden set schema and loader"
```

---

### Task 12: Retrieval metrics

**Files:**
- Create: `src/legalrag/eval/__init__.py`
- Create: `src/legalrag/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float`, `precision_at_k(ranked, relevant, k) -> float`, `reciprocal_rank(ranked, relevant) -> float`, `ndcg_at_k(ranked, relevant, k) -> float`, `RetrievalMetrics` (frozen dataclass with `recall: dict[int, float]`, `precision: dict[int, float]`, `mrr: float`, `ndcg: dict[int, float]`), `aggregate(ranked_lists: Sequence[Sequence[str]], relevant_sets: Sequence[Collection[str]], ks: Sequence[int]) -> RetrievalMetrics`.

- [ ] **Step 1: Write the failing test**

```python
import math

import pytest

from legalrag.eval.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANT = {"a", "b", "c"}


def test_recall_at_k_counts_a_fraction_of_all_relevant():
    ranked = ["a", "x", "y", "z", "b"]
    assert recall_at_k(ranked, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, RELEVANT, 2) == pytest.approx(1 / 3)


def test_precision_at_k_divides_by_k_not_by_relevant_count():
    assert precision_at_k(["a", "x"], RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(["a", "x", "y", "z", "b"], RELEVANT, 5) == pytest.approx(0.4)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["x", "y", "b"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], RELEVANT) == 1.0


def test_ndcg_at_k_matches_hand_computation():
    # relevant at ranks 1 and 3, three relevant total
    ranked = ["a", "x", "b"]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, RELEVANT, 3) == pytest.approx(dcg / idcg)


@pytest.mark.parametrize(
    "func",
    [
        lambda r: recall_at_k(r, RELEVANT, 5),
        lambda r: precision_at_k(r, RELEVANT, 5),
        lambda r: reciprocal_rank(r, RELEVANT),
        lambda r: ndcg_at_k(r, RELEVANT, 5),
    ],
)
def test_empty_retrieval_scores_zero_without_nan(func):
    value = func([])
    assert value == 0.0
    assert not math.isnan(value)


def test_aggregate_averages_across_questions():
    metrics = aggregate([["a", "x"], ["b", "c"]], [{"a"}, {"b", "c"}], ks=[2])
    assert metrics.recall[2] == pytest.approx(1.0)
    assert metrics.precision[2] == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.eval.metrics'`

- [ ] **Step 3: Implement the metric functions**

All take a ranked list of chunk ids and a set of relevant ids. Relevance is binary. `recall_at_k` is `|relevant ∩ ranked[:k]| / |relevant|`. `precision_at_k` is `|relevant ∩ ranked[:k]| / k`. `reciprocal_rank` is `1 / position` of the first relevant id, `0.0` when absent. `ndcg_at_k` uses the standard `1 / log2(rank + 1)` gain with the ideal DCG computed from `min(len(relevant), k)` hits.

This is Review Focus #1 and #3: every function returns `0.0` for an empty ranked list or an empty relevant set, never `NaN` and never raising `ZeroDivisionError`; and when a question has several relevant ids, all of them count. `aggregate` computes each metric per question, weighted by `1 / len(relevant_sets[q])` in the recall numerator only through `recall_at_k` itself, then takes the arithmetic mean across questions, guarding the zero-question case.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/legalrag/eval tests/test_metrics.py
git commit -m "feat: add retrieval metrics"
```

---

### Task 13: Evaluation runner and report

**Files:**
- Create: `src/legalrag/eval/runner.py`
- Create: `src/legalrag/eval/report.py`
- Test: `tests/test_runner.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `GoldenQuestion`, `GoldenSetError` from Task 11; `Retriever` from Task 10; metrics from Task 12.
- Produces: `QuestionResult` (frozen dataclass: `qid: str`, `retrieved_ids: tuple[str, ...]`, `relevant_ids: tuple[str, ...]`, `hit: bool`), `EvalRun` (frozen dataclass: `config: str`, `ks: tuple[int, ...]`, `metrics: RetrievalMetrics`, `results: tuple[QuestionResult, ...]`), `validate_golden_against_corpus(questions, known_chunk_ids: Collection[str]) -> None`, `run_eval(retriever: Retriever, questions: Sequence[GoldenQuestion], *, config: str, k_max: int = 20, ks: Sequence[int] = (5, 10, 20)) -> EvalRun`, `render_report(runs: Sequence[EvalRun]) -> str`, `write_report(markdown: str, path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.runner import EvalRun, run_eval, validate_golden_against_corpus
from legalrag.golden import GoldenQuestion, GoldenSetError
from legalrag.models import Chunk, Retrieved

CORPUS_IDS = {"d#0000", "d#0001", "d#0002"}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, k=10):
        return [
            Retrieved(chunk=Chunk(cid, "d", i, cid, cid, {}), score=1.0 - i * 0.1, rank=i + 1)
            for i, cid in enumerate(self.ids[:k])
        ]


def _question(qid="q001", relevant=("d#0000",)):
    return GoldenQuestion(qid, "q", tuple(relevant), "single_clause", True, None)


def test_validate_raises_on_missing_chunk_id():
    question = _question(relevant=("d#9999",))
    with pytest.raises(GoldenSetError, match="d#9999"):
        validate_golden_against_corpus([question], CORPUS_IDS)


def test_validate_skips_unanswerable_questions():
    question = GoldenQuestion("q002", "q", (), "unanswerable", False, None)
    validate_golden_against_corpus([question], CORPUS_IDS)


def test_run_eval_scores_a_perfect_retriever():
    run = run_eval(FakeRetriever(["d#0000", "d#0002", "d#0001"]), [_question()], config="dense", ks=[5])
    assert isinstance(run, EvalRun)
    assert run.metrics.recall[5] == 1.0
    assert run.metrics.mrr == 1.0
    assert run.results[0].hit is True


def test_run_eval_scores_a_miss():
    run = run_eval(FakeRetriever(["d#0001"]), [_question()], config="dense", ks=[5])
    assert run.metrics.recall[5] == 0.0
    assert run.results[0].hit is False


def test_run_eval_on_no_questions_returns_zero_metrics():
    run = run_eval(FakeRetriever([]), [], config="dense", ks=[5])
    assert run.metrics.mrr == 0.0
    assert run.results == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.eval.runner'`

- [ ] **Step 3: Implement the runner**

`validate_golden_against_corpus` checks every id in `relevant_chunk_ids` against the known corpus ids and raises `GoldenSetError` naming the question and the missing id — this is Review Focus #2, so a corpus change fails loudly instead of silently scoring zero. Unanswerable questions have no relevant ids and are skipped.

`run_eval` validates first, then retrieves `k_max` results per question, builds `QuestionResult` records, and aggregates with `aggregate`. `hit` is true when any relevant id appears in the retrieved list.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the failing report test**

```python
from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.report import render_report
from legalrag.eval.runner import EvalRun


def _run(config, recall5):
    metrics = RetrievalMetrics(
        recall={5: recall5, 10: recall5}, precision={5: recall5 / 2, 10: recall5 / 4},
        mrr=recall5, ndcg={5: recall5, 10: recall5},
    )
    return EvalRun(config=config, ks=(5, 10), metrics=metrics, results=())


def test_render_report_lists_each_configuration():
    markdown = render_report([_run("dense-fixed", 0.62), _run("dense-structured", 0.71)])
    assert "dense-fixed" in markdown
    assert "dense-structured" in markdown
    assert "0.62" in markdown
    assert "0.71" in markdown
    assert markdown.count("|") > 8


def test_render_report_handles_no_runs():
    assert "dense" not in render_report([])
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.eval.report'`

- [ ] **Step 7: Implement `render_report` and `write_report`**

One markdown table with a row per `EvalRun` and columns `config`, `recall@5`, `recall@10`, `recall@20`, `precision@5`, `mrr`, `ndcg@10`. Format every metric to two decimal places. Prepend a `# Retrieval evaluation` heading and a line stating the number of configurations and the question count taken from the first run. Return an empty string when `runs` is empty. `write_report` writes UTF-8 with `\n` line endings.

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add src/legalrag/eval/runner.py src/legalrag/eval/report.py tests/test_runner.py tests/test_report.py
git commit -m "feat: add evaluation runner and markdown report"
```

---

### Task 14: Evaluation CLI, Makefile and CI

**Files:**
- Create: `src/legalrag/eval/__main__.py`
- Create: `Makefile`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 13.
- Produces: `main(argv: list[str] | None = None) -> int` in `legalrag/eval/__main__.py`, runnable as `python -m legalrag.eval`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from legalrag.eval.__main__ import main

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def test_eval_cli_writes_results_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["--ingest-fixtures", str(FIXTURES), "--golden", "golden.yaml"]) == 0
    assert "recall@5" in (tmp_path / "results.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.eval.__main__'`

- [ ] **Step 3: Implement the evaluation CLI**

`main` accepts `--golden PATH` (default `data/golden/milestone1.yaml`), `--config NAME` (default `dense-fixed`), `--collection NAME`, `--k-max N` (default 20), `--report PATH` (default `results.md`), and `--ingest-fixtures PATH`.

Default flow: read `{data_dir}/chunks/chunks.jsonl`, rebuild the index with `recreate()` and fresh embeddings, load the golden set, validate it against the corpus, run one `EvalRun` per configuration (only `dense-fixed` exists at this milestone), render and write the report. Print the table to stdout. Return 0 on success and 1 on `GoldenSetError`, printing the message.

`--ingest-fixtures PATH` runs the Task 7 ingest path against a local snapshot directory first, which is what keeps the test offline.

The test writes its own tiny two-question `golden.yaml` into `tmp_path` before calling `main`, referencing chunk ids it derives from the ingest output. Add a `Makefile` with `test` (`python -m pytest -m "not slow"`), `test-all`, `ingest`, `eval` (`python -m legalrag.eval`) and `.PHONY`. Add `.github/workflows/ci.yml` running on `ubuntu-latest` with Python 3.11: install with `pip install -e ".[dev]"`, run `python -m pytest -m "not slow"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the whole non-slow suite**

Run: `python -m pytest -m "not slow" -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/legalrag/eval/__main__.py Makefile .github tests/test_eval_cli.py
git commit -m "feat: add evaluation CLI, makefile and CI"
```

---

### Task 15: Populate the 20-question golden set

**Files:**
- Modify: `data/golden/milestone1.yaml`
- Create: `src/legalrag/golden_candidates.py`
- Test: `tests/test_golden_candidates.py`

**Interfaces:**
- Consumes: `Chunk` from Task 2; `GoldenQuestion` from Task 11.
- Produces: `sample_chunks(chunks: Sequence[Chunk], n: int, *, seed: int = 0) -> list[Chunk]` and `render_candidates(chunks: Sequence[Chunk]) -> str`.

This task is mostly manual work, which is why its check is a command rather than a unit test.

- [ ] **Step 1: Implement the candidate sampler and its test**

```python
from legalrag.golden_candidates import render_candidates, sample_chunks
from legalrag.models import Chunk


def test_sample_chunks_is_deterministic_and_bounded():
    chunks = [Chunk(f"d#{i:04d}", "d", i, f"text {i}", f"text {i}", {}) for i in range(50)]
    first = sample_chunks(chunks, 10, seed=7)
    assert first == sample_chunks(chunks, 10, seed=7)
    assert len(first) == 10
    assert all(c in chunks for c in first)


def test_render_candidates_includes_chunk_ids_and_text():
    chunks = [Chunk("d#0000", "d", 0, "alpha text", "alpha text", {})]
    rendered = render_candidates(chunks)
    assert "d#0000" in rendered
    assert "alpha text" in rendered
```

`sample_chunks` is a seeded `random.Random(seed).sample`. `render_candidates` emits one fenced block per chunk with `chunk_id`, `citation` from metadata, and the full text — the input to a language model that proposes one question per chunk.

- [ ] **Step 2: Ingest the real corpus**

Run: `python -m legalrag.ingest --source rbi`
Expected: prints a document count, chunk count and parse-failure count. Parse failures are reported, not fatal.

- [ ] **Step 3: Generate candidates and verify them by hand**

Run: `python -c "from legalrag.corpus import read_chunks; from legalrag.golden_candidates import sample_chunks, render_candidates; print(render_candidates(sample_chunks(list(read_chunks('data/chunks/chunks.jsonl')), 60, seed=1)))"`

Paste the output into a language model session and ask for one question per chunk plus the chunk id that answers it. Then **verify every proposal by reading the chunk**: keep, edit, or discard. This is language-model-assisted generation with human verification, and the README must say so.

Fill `data/golden/milestone1.yaml` with the 20 survivors, meeting the mix in the spec: roughly 10 `single_clause`, 4 `multi_clause`, 2 `cross_document`, 2 `temporal`, 2 `unanswerable`. `relevant_chunk_ids` must contain the chunk ids you verified, and unanswerable questions must have an empty list. Add `data/golden/README.md` with the one-page annotation guide: what counts as a correct chunk, what makes a question unanswerable, and how granular a citation must be.

- [ ] **Step 4: Run the full evaluation**

Run: `python -m legalrag.eval`
Expected: `results.md` is written and prints a table containing a `dense-fixed` row with non-zero `recall@5`, `recall@10` and `mrr`, and a stated question count of 20. A validation error naming a missing chunk id means a golden entry does not correspond to the corpus — fix the entry, do not weaken the check.

- [ ] **Step 5: Record the baseline in the README**

Create `README.md` stating the goal in two sentences, the corpus and its sources, how to run ingest and eval, and the baseline table copied from `results.md`. Note the known limits plainly: retrieval only, no answer generation yet, no temporal filtering yet.

- [ ] **Step 6: Commit**

```bash
git add data/golden README.md src/legalrag/golden_candidates.py tests/test_golden_candidates.py results.md
git commit -m "feat: add verified 20-question golden set and baseline results"
```

---

### Task 16: Retrieval inspector

**Files:**
- Create: `src/legalrag/inspector.py`
- Modify: `pyproject.toml` (add `streamlit` to the `dev` extra)
- Modify: `Makefile` (add an `inspect` target)
- Test: `tests/test_inspector.py`

**Interfaces:**
- Consumes: `ChunkIndex` from Task 9; `Embedder` from Task 8; `DenseRetriever` from Task 10; `Retrieved` from Task 2; `Settings` from Task 1.
- Produces: `results_rows(hits: Sequence[Retrieved]) -> list[dict]`.

This is a diagnostic tool, not a product surface. It exists so retrieval quality can be inspected by eye while the ablation work is being done. The deployed product surface is a later piece of work, after answer generation exists.

The pure data-shaping function is separated from the rendering on purpose: Streamlit rendering is not worth unit-testing, but the table contents are.

- [ ] **Step 1: Write the failing test**

```python
from legalrag.inspector import results_rows
from legalrag.models import Chunk, Retrieved


def test_results_rows_shapes_hits_for_display():
    hits = [
        Retrieved(
            chunk=Chunk("d#0000", "d", 0, "text zero", "text zero", {"citation": "CGST Rules, 2017, Rule 36(4)"}),
            score=0.91,
            rank=1,
        ),
        Retrieved(chunk=Chunk("d#0001", "d", 1, "text one", "text one", {}), score=0.42, rank=2),
    ]
    rows = results_rows(hits)
    assert rows[0]["rank"] == 1
    assert rows[0]["chunk_id"] == "d#0000"
    assert rows[0]["citation"] == "CGST Rules, 2017, Rule 36(4)"
    assert rows[0]["text"] == "text zero"
    assert list(rows[0]) == ["rank", "score", "chunk_id", "doc_id", "citation", "text", "chars"]


def test_results_rows_handles_no_hits():
    assert results_rows([]) == []


def test_results_rows_uses_empty_citation_when_metadata_is_missing():
    hits = [Retrieved(chunk=Chunk("d#0000", "d", 0, "t", "t", {}), score=1.0, rank=1)]
    assert results_rows(hits)[0]["citation"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inspector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legalrag.inspector'`

- [ ] **Step 3: Implement `results_rows` and the Streamlit page**

`results_rows` maps each `Retrieved` to a row dict with `rank`, `score` rounded to four decimals, `chunk_id`, `doc_id`, `citation` read from `chunk.metadata.get("citation", "")`, `text`, and `chars` as `len(chunk.text)`.

The Streamlit page, run via `python -m legalrag.inspector`, loads settings, reads `{data_dir}/chunks/chunks.jsonl`, rebuilds `ChunkIndex` with `recreate()`, builds a `DenseRetriever`, and renders a text input for the query, a number input for `k` (default 10, maximum 50), and the results as a dataframe from `results_rows` with the full chunk text in an expander per row. Show the index chunk count in the sidebar. Add an `inspect` target to the `Makefile` running `python -m legalrag.inspector`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inspector.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Verify the page renders against the real corpus**

Run: `make inspect`
Expected: the app starts and reports a non-zero chunk count in the sidebar. Run a query from the golden set and confirm the verified chunk appears in the returned rows. Stop the app.

- [ ] **Step 6: Commit**

```bash
git add src/legalrag/inspector.py pyproject.toml Makefile tests/test_inspector.py
git commit -m "feat: add retrieval inspector"
```

---

## Notes for the Implementer

- Tasks 1 to 14 are all offline and deterministic except the one-time fixture capture in Task 4 and the model download in Task 8. Only Task 15 touches the live network at scale.
- `results.md` is committed. It is a checked-in artifact, not a build output. Re-running the evaluation and committing the change is how each later milestone shows its improvement.
- When the corpus is re-ingested and chunk ids change, `validate_golden_against_corpus` will fail. That is the intended behaviour. Re-verify the affected questions rather than regenerating the whole set.
- Nothing in this plan calls a language model at runtime. The one place a model is used is Task 15, by hand, to propose candidate questions.
