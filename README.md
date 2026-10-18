# india-legal-rag

A retrieval system over Indian regulatory text, built so that its quality can be
measured rather than asserted. Every change is judged by a scored, versioned
golden set, and the evaluation report is committed alongside the code.

This milestone is the retrieval baseline: corpus, embeddings, vector search, an
evaluation harness and an inspection tool. There is no answer generation yet.

## Status

| milestone | scope | state |
| --- | --- | --- |
| 1 | corpus, dense retrieval baseline, evaluation harness | this milestone |
| 2 | answer generation, citations, refusal, model-judge calibration | not started |

## Corpus

25 notifications from the Reserve Bank of India, captured on 2026-10-18 and
stored as HTML snapshots under `data/raw/rbi/`. Ingest normalises each page to
plain text, drops the site's navigation and archive markup, and splits the body
into fixed-size overlapping chunks of about 2,000 characters. The result is 154
chunks with source URL, circular number and publication date attached. 25 of 25
documents parse, with no failures.

Sources:

- RBI notifications index: <https://www.rbi.org.in/Scripts/NotificationUser.aspx>

Other Indian regulatory sources (CBIC for GST, India Code for Acts) are planned,
not present.

## Running it

Requires Python 3.11. On Windows:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# fetch and index the corpus (network)
.venv\Scripts\python.exe -m legalrag.ingest --source rbi

# or ingest the committed snapshots, offline
.venv\Scripts\python.exe -m legalrag.ingest --source rbi --snapshots tests/fixtures/rbi

# score retrieval against the golden set, writing results.md
.venv\Scripts\python.exe -m legalrag.eval

# open the retrieval inspector
.venv\Scripts\python.exe -m legalrag.inspector
```

`make test`, `make test-all`, `make ingest`, `make eval` and `make inspect` wrap
the same commands.

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about
130 MB) and embeds on CPU. No API key is needed and nothing calls a hosted
language model at runtime.

## Baseline

Dense bi-encoder retrieval over the whole corpus, no reranking, no hybrid search.

| config | recall@5 | recall@10 | recall@20 | precision@5 | mrr | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- |
| dense-fixed | 0.78 | 0.82 | 0.85 | 0.22 | 0.73 | 0.73 |

Measured over 20 questions. `results.md` is the committed output of
`python -m legalrag.eval` and is rewritten by each run.

Two caveats on reading these numbers:

- Two of the twenty questions are unanswerable by design, and score zero on
  every retrieval metric, so recall@5 cannot exceed 0.90 on this set. The
  figures therefore understate quality on answerable questions.
- One question is not retrieved at all, at any depth up to 20: `q008`, which
  asks for the Differential Rate of Interest Scheme's amount and rate of
  interest. The relevant chunk exists and is labelled. That is a genuine
  baseline miss, and the kind of thing the later ablation work exists to fix.

## The golden set

`data/golden/milestone1.yaml` holds 20 questions with the chunk ids that answer
them, mixed across five categories: 10 single-clause, 4 multi-clause, 2
cross-document, 2 temporal and 2 unanswerable.

Questions were proposed with language-model assistance from the ingested chunk
text and then checked: every labelled chunk was confirmed to contain the fact it
is labelled for, and every labelled chunk id was confirmed present in the
corpus. Judging whether a question is fair and phrased naturally remains a human
review step. `data/golden/README.md` is the annotation guide, covering what
counts as a correct chunk, what makes a question unanswerable, and how granular
a citation must be.

## Known limits

- Retrieval only. No generated answers, no answer-level citations, no refusal.
- No temporal filtering. A question about the current position may retrieve a
  superseded circular; both `temporal` questions happen to retrieve correctly
  today, which is luck, not a mechanism.
- Dense-only. No lexical retrieval, no fusion, no reranking.
- Fixed-size chunking, which splits provisions across boundaries.
- RBI notifications only. Nothing from CBIC, India Code, SEBI or the Income Tax
  Department, so nothing outside this corpus is answerable.
- English only. Hindi and regional-language content is out of scope for now.
- Embeddings run on CPU. The index is local Qdrant on disk; there is no server
  deployment.

## Documentation

- `docs/decisions.md` — decision log, one entry per settled question.
- `docs/architecture-changelog.md` — how the architecture changed and why.
- `docs/glossary.md` — terms as they are used in this repository.
- `docs/troubleshooting.md` — environment problems and their fixes.
- `docs/README.md` — index of the documentation set.
- `docs/superpowers/specs/` — the design this milestone implements.
- `docs/superpowers/plans/` — the task-by-task implementation plan.

## Layout

```
src/legalrag/
  config.py       settings
  models.py       Document, Chunk, Retrieved
  normalize.py    text normalisation
  sources/rbi.py  fetch, parse and index RBI notifications
  chunk.py        fixed-size overlapping chunking
  corpus.py       JSONL document and chunk stores
  ingest.py       ingest command line
  embed.py        sentence-transformers wrapper
  index.py        Qdrant chunk index
  retrieve.py     dense retriever
  golden.py       golden set schema and loader
  eval/           metrics, runner, report, command line
  inspector.py    Streamlit retrieval inspector
```
