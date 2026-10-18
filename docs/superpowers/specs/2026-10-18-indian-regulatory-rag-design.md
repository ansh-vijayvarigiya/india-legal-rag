# Eval-First RAG over Indian Regulatory Text — Design

Date: 2026-10-18
Status: Approved design. Implementation plan not yet written.

## 1. Goal

Build one portfolio project that supports a move from generalist software engineering into a
GenAI / LLM application engineer role in the Indian market (target compensation around 25 LPA).

The differentiator is deliberately not "a RAG application". Anyone can glue an embedding model to
a vector store. The scarce skill is proving a retrieval system works, quantifying each design
choice, and explaining the failures. So the deliverable is a working RAG system plus a measured
evaluation harness and an ablation table.

Success criteria:

1. `make eval` runs end to end and writes `results.md` containing a row per retrieval
   configuration and a column per metric.
2. Later configurations beat the naive baseline on retrieval metrics, with recorded deltas.
3. A 100-question golden set exists, is human-verified, and is versioned in the repository.
4. Ground-truth chunk labels are derived from source text that any careful reader can verify, so
   the project does not require domain expertise to maintain.
5. The README presents the results table, a failure analysis of the worst cases, and the reasoning
   behind each design decision.
6. Unit tests pass in CI, and a reduced evaluation smoke test runs in CI.
7. Every answer cites the specific clause it came from, and the system refuses to answer when
   retrieval confidence is low.

## 2. Non-goals

Deferred, not rejected. Each is listed so implementation does not creep into it.

- Web UI, authentication, user accounts, billing.
- Agentic or multi-step planning, tool use, agent orchestration.
- Model fine-tuning.
- Hindi or other Indian-language content. The corpus is English only.
- Multi-tenant or hosted production deployment. The service runs locally.

## 3. Corpus

Three public sources, all with stable official URLs:

- **India Code** (`https://www.indiacode.nic.in`) — a DSpace repository, organised into collections
  for Acts, Rules, Regulations, Notifications, Circulars, Statutes, Orders and Ordinances.
  Carries structured metadata: enactment date, act number, short title, ministry, department,
  notification number.
- **Reserve Bank of India** (`https://www.rbi.org.in`) — notification index and per-notification
  pages. Carries circular number, issue date, department, subject and target audience. Full text is
  available as HTML in addition to PDF.
- **CBIC GST** (`https://cbic-gst.gov.in`) — circulars, notifications, orders, and consolidated
  PDFs of the CGST Rules.

Initial slice, kept narrow on purpose:

- Central Goods and Services Tax Act, 2017.
- Central Goods and Services Tax Rules, 2017, including the consolidated amended version.
- GST circulars and notifications, 2017 onward.
- RBI: Foreign Exchange Management Act material, and the Master Direction on Know Your Customer.

This is on the order of a few thousand pages: substantial enough that retrieval is genuinely hard,
small enough that ground truth can be labelled by hand.

Government legislation and regulatory circulars are public documents. The repository records the
source URL for every ingested document so attribution is always available.

## 4. Ingestion

Pipeline stages, each producing an artifact that can be inspected and re-run:

```
fetch -> parse -> normalise -> chunk -> enrich -> embed -> index
```

- **Fetch.** One fetcher per source. Requests are rate limited and identify the client. Raw
  responses (HTML and PDF) are written to `data/raw/` and committed as a fixture set for a small
  sample. Re-fetching is idempotent and keyed on a content hash.
- **Parse.** `pymupdf` for PDF text and layout, a standard HTML parser for RBI pages. Parsed output
  is written to `data/parsed/` as a normalised document record: source URL, document identifier,
  title, publication date, and an ordered list of text blocks with page anchors.
- **Normalise.** Whitespace, ligatures, hyphenation across line breaks, and the common gazette
  artefacts. Amendment footnotes are preserved but tagged, since they carry version information.
- **Chunk.** Structure-aware, described in section 5.
- **Enrich.** Metadata attached per chunk, described in section 6.
- **Embed and index.** Described in section 7.

Ingestion is a command, not a notebook. The repository is `india-legal-rag` and the Python package
is `legalrag`, so re-ingesting one source end to end is a single command:
`python -m legalrag.ingest --source <name>`. Each of the stages above is separately runnable against
the artifacts the previous stage wrote.

## 5. Chunking

Fixed-size chunking is the naive baseline and is retained only as retrieval configuration 0.

The production chunker is structure-aware. Indian legislation has a real hierarchy:

```
Act -> Chapter -> Section -> sub-section -> clause -> proviso / explanation
Rules -> Rule -> sub-rule -> clause
Circular -> paragraph
```

Rules:

- Split on the strongest available structural boundary, then merge adjacent units only while the
  result stays under the model's context budget. Never split mid-clause.
- Every chunk carries its full structural path as metadata, for example
  `CGST Act 2017 > Chapter V > Section 16 > sub-section (2) > clause (b)`.
- A chunk also carries a short "breadcrumb" prefix that is prepended to the embedded text, so an
  isolated clause still reads in context. The stored text is the original clause; the breadcrumb is
  an embedding input, not part of the cited evidence.
- Overlap is applied only at the merged-unit boundary, and only when the merge crosses a boundary
  that carries semantic weight.

This is the first meaningful design decision and the first row of the ablation table.

## 6. Metadata and the temporal model

Per-chunk metadata:

- `act` / `instrument` — the parent instrument.
- `citation` — human-readable, for example `CGST Rules, 2017, Rule 36(4)`.
- `structural_path` — as above.
- `source_url`
- `page`
- `effective_from`, `effective_to`
- `amendment_chain` — list of amending notifications, each with an identifier and date.
- `superseded_by` — identifier of the replacing provision, when known.
- `content_hash`

Law changes. A question about a tax position in a given year must not be answered with the current
text if the text changed. The temporal model is the project's strongest differentiator, because most
retrieval systems ignore it:

- Amendment links are extracted from footnote and amendment markers in the source text, which are
  already tagged during normalisation.
- Where a consolidated "as amended" version exists, it is ingested as a dated version rather than
  replacing earlier text. Each version becomes its own set of chunks sharing a `citation`.
- Retrieval accepts an optional `as_on` date and filters to chunks whose effective interval
  contains that date. With no date supplied, the current version is preferred and the response
  states which version was used.

Extraction of the amendment chain from gazette footnotes is heuristic. Where a link cannot be
resolved confidently it is recorded as unresolved rather than guessed, and the count of unresolved
links is reported alongside the evaluation results.

## 7. Retrieval

A ladder of configurations. Each rung is evaluated, and each rung must earn its place.

| # | Configuration |
|---|---|
| 0 | Dense retrieval only, fixed-size chunks, no metadata |
| 1 | Structure-aware chunks with metadata paths |
| 2 | Hybrid: BM25 lexical fused with dense using Reciprocal Rank Fusion |
| 3 | Cross-encoder reranking of the top 50 candidates |
| 4 | Query transformation: acronym expansion and legal-citation normalisation |
| 5 | Temporal filtering by `as_on` date |

Components:

- **Dense store:** Qdrant in local on-disk mode, chosen because Docker is not available on the
  development machine. The client API is isolated behind one module so the same code runs against a
  Qdrant server later without touching retrieval logic.
- **Embeddings:** a local `BAAI/bge` model via `sentence-transformers`. Runs on the available GPU.
  Embedding choice is itself an ablation dimension: `bge-small-en-v1.5` against
  `bge-base-en-v1.5`, both run locally, so there is no embedding API cost.
- **Lexical:** BM25 over the same chunks, using a pure-Python implementation with a legal-aware
  tokenizer written here. Tokenisation control matters more than index persistence for this corpus,
  because citations such as `36(4)` and acronyms such as `ITC` must be indexed as single terms.
- **Fusion:** Reciprocal Rank Fusion implemented in this repository. It is roughly fifteen lines and
  is worth owning, since it has to be explained in an interview.
- **Reranker:** a local `BAAI/bge-reranker` cross-encoder over the top 50 candidates.
- **Query transformation:** rules plus a small model call. Acronym expansion matters because legal
  text is dense with acronyms; a query using "ITC" and a statute using "input tax credit" must meet
  in the middle.
- **Answering:** a large language model produces a response grounded strictly in the retrieved
  chunks and returns clause-level citations. The model interface is isolated in a single module so
  the provider and model are a one-line change. A local Ollama model is the fallback when no hosted
  API key is configured, with the understanding that judge quality is lower.

Two behaviours are treated as first-class, because both are measurable:

- **Citation validity** — every citation in an answer maps to a chunk that was actually retrieved.
- **Refusal** — when retrieval confidence is below a threshold, the system says it cannot answer
  from the corpus instead of guessing. Refusal correctness is measured on the unanswerable portion
  of the golden set.

## 8. Evaluation harness

This is the spine of the project and is built second, immediately after the naive baseline, so that
every subsequent change has a before and after.

**Retrieval metrics.** No language model involved, so they are cheap, objective and stable:
recall@k, precision@k, mean reciprocal rank, and nDCG@10.

**Answer metrics:**

- *Faithfulness* — every claim in the answer appears in a retrieved chunk.
- *Correctness* — the answer matches the golden answer.
- *Citation validity* — mechanical, described above.
- *Refusal correctness* — measured on unanswerable questions: refusal is correct, an invented
  answer is not.

Answer scoring uses a language model as judge, with a written rubric. The judge is calibrated
against roughly twenty manually scored examples, and the measured agreement between the judge and
the manual labels is reported with the results. A judge that has not been calibrated is not
evidence.

**Operational metrics.** p50 and p95 latency, token counts, and cost per query, reported per
configuration. The cost of each rung is part of the argument: a rung that improves recall by two
points at tripled latency is a different decision from a free win.

**Output.** A generated `results.md` containing the configuration-by-metric table, the judge
agreement number, the unresolved-amendment-link count, and a failure analysis listing the ten worst
cases with a short explanation of why each failed.

**Reproducibility.** One command runs the whole evaluation. Language model responses are cached on
disk keyed by prompt and model, so re-running is cheap and results are stable across runs. Seeds are
fixed for anything stochastic.

## 9. Golden set construction

The labelling burden is the main risk to this project, so it is removed by design. Two different
kinds of label exist, and only one of them needs domain knowledge.

- **Retrieval ground truth: which chunks contain the answer.** Establishing this requires reading a
  clause and confirming that it is where the provision is stated. That is reading comprehension, not
  tax expertise.
- **Answer ground truth: the correct final answer.** Harder. However, for legislation the answer is
  usually stated verbatim in the clause, so it is graded by citation match plus the calibrated
  judge.

Construction method:

1. **Reverse generation.** Walk the chunks; a language model proposes "what question does this chunk
   answer?" Each proposal is verified by reading the chunk and keeping, editing or discarding it.
   This is standard practice and is disclosed as *language-model-assisted generation with human
   verification*.
2. **Hand-written hard cases.** The categories where generation cannot be trusted are written by
   hand: unanswerable questions, temporal questions, and cross-document questions.
3. **Target mix:** approximately 50 single-clause, 20 multi-clause, 10 cross-document, 10 temporal,
   10 unanswerable.
4. **Annotation guide.** A one-page document defining what counts as a correct chunk, what makes a
   question unanswerable, and how granular a citation must be. It is reused whenever questions are
   added.
5. **Label stability.** Twenty questions are re-verified a week later and the self-agreement rate is
   reported. The golden set is versioned, and changes to it are reviewed like code.

Around 200 candidates are generated and the 100 that pass verification are kept. Expected effort is
three to four hours. The set can grow later, since it is versioned.

## 10. Service surface

A thin FastAPI application, which is the part of the project that reuses existing strengths:

- `POST /search` — retrieval only, returns chunks with scores and metadata. Used by the evaluation
  harness.
- `POST /answer` — grounded answer with citations, streamed.
- `GET /eval/summary` — the latest results table, machine-readable.
- `GET /health`

Request and response models are Pydantic. Configuration is environment-driven. The service is thin
on purpose: it exposes the system, it is not the system.

## 11. Observability

Every query produces a trace record: query text, configuration used, retrieved chunks with scores,
final answer, citations, token counts, latency, and cost. Records are written to a local SQLite
database, which needs no infrastructure and doubles as the source for the operational metrics in
`results.md`. A JSONL export is provided so traces can be loaded into a hosted tracing tool later.

## 12. Testing

- **Unit tests** for the chunker, with explicit clause-boundary cases; the parsers, using committed
  fixtures taken from a small sample of real documents; Reciprocal Rank Fusion; metadata filtering;
  temporal filtering; and the citation validator.
- **Evaluation harness tests.** With a fake retriever and fake judge, the metric computations must
  produce known-correct values. Metrics that are not themselves tested are not trustworthy.
- **CI.** GitHub Actions runs the unit tests and a reduced evaluation smoke test over a
  twenty-question subset of the golden set.
- **Deliberately not tested:** language model answer quality. That is the evaluation harness's job,
  not a unit test's, and asserting on generated text produces brittle tests that fail for unrelated
  reasons.

## 13. Milestones

Each milestone ends in something that can be demonstrated.

1. Naive end-to-end pipeline (dense retrieval, fixed chunks) plus a twenty-question golden set and a
   working evaluation runner. Output: baseline numbers.
2. Structure-aware chunking and metadata. Output: first measured improvement.
3. Hybrid retrieval with Reciprocal Rank Fusion, then cross-encoder reranking. Output: two more rows
   in the table.
4. Query transformation and temporal filtering. Output: the remaining rows, plus the temporal
   behaviour demonstrated.
5. FastAPI service, trace logging, and cost and latency reporting.
6. README with the results table, failure analysis, and design decisions.

A later, separate piece of work may add a user interface, authentication and deployment.

## 14. Environment

- Windows (win32), PowerShell.
- Python 3.11.9 is the interpreter on `PATH`; 3.14 is also installed. The project pins 3.11 in a
  virtual environment.
- git 2.55 is available.
- Docker is **not** installed. This is the reason for Qdrant local mode and SQLite rather than
  Postgres with pgvector and a self-hosted tracing tool.
- GPU: NVIDIA RTX PRO 500 (Blackwell), 6 GB VRAM. 16 logical cores and 32 GB RAM.
- PyTorch support for Blackwell requires a recent build with CUDA 12.8 wheels. If that installation
  proves troublesome, everything runs on CPU, which is acceptable for a corpus of this size.

## 15. Risks

| Risk | Response |
|---|---|
| PDF parsing quality on gazette documents — tables, footnotes, amendment markers | Fixture-based parser tests on real documents; unresolved amendments are counted and reported rather than guessed |
| Amendment chain extraction is heuristic | Links are recorded as unresolved when uncertain, and the unresolved count appears next to the results |
| PyTorch and CUDA on Blackwell may need a specific build | CPU fallback is acceptable for this corpus size |
| Language model API access and a small budget are needed for answering and judging | Model access is isolated in one module; a local model is the fallback |
| Scraping may be fragile if source pages change | Raw snapshots are stored so evaluations reproduce against a fixed corpus |
| Labelling effort could expand | The reverse-generation method with a target mix bounds the work; the set is versioned and can grow later |
