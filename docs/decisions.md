# Architecture decisions

Decisions that shaped this project, in the order they were made. Each entry records what was
decided, why, what was rejected, and what it costs if the call turns out wrong.

| # | Decision | Date |
|---|---|---|
| D-001 | Ingest RBI first; defer India Code and CBIC | 2026-10-18 |
| D-002 | Retrieval metrics before answer metrics | 2026-10-18 |
| D-003 | Split spec milestone 1 into two implementation plans | 2026-10-18 |
| D-004 | Qdrant in local on-disk mode instead of Postgres with pgvector | 2026-10-18 |
| D-005 | SQLite for trace records instead of a hosted tracing tool | 2026-10-18 |
| D-006 | Local `bge` embeddings and reranker; no embedding API | 2026-10-18 |
| D-007 | Implement Reciprocal Rank Fusion here rather than use a library | 2026-10-18 |
| D-008 | Pure-Python BM25 with a legal-aware tokenizer instead of SQLite FTS5 | 2026-10-18 |
| D-009 | No vLLM; Ollama only if a local model is ever needed | 2026-10-18 |
| D-010 | A hosted free-tier model judges answers, not a local model | 2026-10-18 |
| D-011 | The Streamlit inspector is a diagnostic tool, not the product surface | 2026-10-18 |
| D-012 | Ground-truth labels come from reverse generation plus human verification | 2026-10-18 |
| D-013 | The repo opens private, and the work email is stripped from git history | 2026-10-18 |
| D-014 | Archify receipt JSON files are not committed | 2026-10-18 |
| D-015 | Emptying the index drops the local storage directory | 2026-10-18 |

---

## D-001 — Ingest RBI first; defer India Code and CBIC (2026-10-18)

**Context.** The spec names three sources: India Code, RBI and CBIC GST. Ingesting all three before
there is any way to measure retrieval quality means building a large pipeline with no feedback.

**Decision.** The first plan ingests RBI only.

**Why.** RBI serves circular _full text as HTML_, so no PDF parsing is needed to reach a real
corpus. That removes the riskiest and slowest stage from the critical path and gets a measurable
baseline sooner.

**Rejected.** All three sources at once, which would have front-loaded PDF parsing and layout
extraction before any retrieval number existed.

**Cost if wrong.** The PDF path stays unexercised longer than ideal, so a PDF-specific problem
surfaces a milestone later than it otherwise would.

## D-002 — Retrieval metrics before answer metrics (2026-10-18)

**Context.** The spec's evaluation harness has two halves: retrieval metrics (recall, precision,
MRR, nDCG) and answer metrics (faithfulness, correctness, citation validity, refusal).

**Decision.** Build and ship the retrieval half first, with no language model involved at all.

**Why.** Retrieval metrics are arithmetic — cheap, deterministic, and stable — and every later
change is measured against them. They also need no API key and cost nothing to run, so work can
start immediately. Answer metrics need a calibrated judge, which is a research-ish task of its own.

**Rejected.** Building the whole harness at once, which would have blocked all progress on judge
calibration.

**Cost if wrong.** A retrieval improvement that makes answers worse would not be caught until the
judge exists. The metric set covers this partially: citation validity and refusal are specified as
first-class for exactly this reason.

## D-003 — Split spec milestone 1 into two implementation plans (2026-10-18)

**Context.** Spec milestone 1 as written covers fetch, parse, chunk, embed, index, retrieval, the
golden set, metrics, the runner, the report, the CLI and CI.

**Decision.** Two plans. Plan A is corpus and retrieval baseline; Plan B is answering, citations,
refusal, the judge and its calibration.

**Why.** Each plan is a coherent, independently testable deliverable. One plan covering everything
would have been enormous and would have made the first shippable result weeks away.

**Rejected.** One large plan, and also splitting by layer rather than by deliverable — a "parsing
plan" or a "metrics plan" would not have produced working software on its own.

**Cost if wrong.** The seam between A and B is the answering layer's interface to the retriever.
The `Retriever` protocol is the boundary, and it exists in Plan A precisely so this seam is a
protocol and not a rewrite.

## D-004 — Qdrant in local on-disk mode instead of Postgres with pgvector (2026-10-18)

**Context.** The development machine has no Docker, and the obvious "serious" choice for a vector
store in a portfolio project is Postgres with the pgvector extension.

**Decision.** Qdrant, in local on-disk mode via `QdrantClient(path=...)`.

**Why.** It is a real vector database, it needs no container or server, and it is a single Python
dependency. All Qdrant calls are confined to one module, so retrieval code can move to a Qdrant
server later without change.

**Rejected.** Installing Postgres and pgvector natively on Windows, which is painful; installing
Docker Desktop, which is a heavier change to the machine than this project warrants; and a
numpy brute-force search, which would not scale and would read as unserious.

**Cost if wrong.** Local mode has fewer features than server mode. If later milestones need
server-only features, that is a configuration change plus a container, not a rewrite.

## D-005 — SQLite for trace records instead of a hosted tracing tool (2026-10-18)

**Context.** The spec calls for tracing every query: retrieved chunks, scores, cost, latency.

**Decision.** Write trace records to a local SQLite database, which also feeds the operational
metrics in `results.md`. Provide a JSONL export.

**Why.** No infrastructure, no account, no API key, and the data is needed in a queryable form for
the report anyway. Owning the trace layer is also more defensible in an interview than clicking a
hosted tool's toggle.

**Rejected.** Self-hosted Langfuse, which needs Docker; Langfuse cloud, which needs an account and
an outbound dependency for a local evaluation loop.

**Cost if wrong.** Less visual polish than a dedicated tracing UI. The JSONL export is the escape
hatch if that matters later.

## D-006 — Local `bge` embeddings and reranker; no embedding API (2026-10-18)

**Context.** Embedding a few thousand pages needs a model. Options were a hosted embedding API or a
local model.

**Decision.** Run `bge-small-en-v1.5` and `bge-base-en-v1.5` locally via `sentence-transformers`,
and `bge-reranker-base` as the cross-encoder. Embedding choice becomes an ablation dimension.

**Why.** The machine has an RTX PRO 500 (Blackwell) with 6 GB VRAM and 32 GB of RAM, which is
plenty for these model sizes. Local embeddings remove an API key, a per-token cost, and a network
dependency from the ingest and evaluation loops.

**Rejected.** A hosted embedding model, which would have added cost and a rate limit to every
experiment re-run.

**Cost if wrong.** Blackwell needs a recent PyTorch build with CUDA 12.8 wheels. If that install
proves troublesome, everything runs on CPU, which is acceptable at this corpus size.

## D-007 — Implement Reciprocal Rank Fusion here rather than use a library (2026-10-18)

**Context.** Hybrid search needs a way to merge a dense ranked list with a lexical ranked list.

**Decision.** Implement RRF in this repository.

**Why.** It is roughly fifteen lines, it is testable as a pure function, and it is a component the
author must be able to explain in an interview. Owning it costs less than owning a dependency plus
its configuration surface.

**Rejected.** Qdrant's native sparse-vector fusion, which hides the ranking decision behind a flag.

**Cost if wrong.** A subtly wrong fusion weighting would be our bug rather than a library's, which
is why the fusion function is unit-tested with hand-computed expected orderings.

## D-008 — Pure-Python BM25 with a legal-aware tokenizer instead of SQLite FTS5 (2026-10-18)

**Context.** Legal text is dense with citations (`36(4)`) and acronyms (`ITC`), and those tokens
must survive tokenization to be searchable.

**Decision.** Pure-Python BM25 with a tokenizer written for this corpus.

**Why.** Tokenization control matters more than index persistence at this scale. FTS5's default
tokenizers split citations in ways that hurt exactly the queries this corpus attracts.

**Rejected.** SQLite FTS5, which would have reused the same database as the trace store but given
up control of tokenization.

**Cost if wrong.** A pure-Python index is rebuilt in memory, which is fine for a few thousand
documents and would need replacing at a much larger corpus size.

## D-009 — No vLLM; Ollama only if a local model is ever needed (2026-10-18)

**Context.** Running a local model was considered, with vLLM proposed.

**Decision.** No vLLM. If a local model is ever needed, use Ollama.

**Why.** vLLM has no native Windows support and requires Linux; the machine's WSL feature is enabled
but has no distribution installed, so vLLM would mean installing Ubuntu, CUDA passthrough and a
second Python environment, on 6 GB of VRAM. Ollama does the same job natively on Windows. Plan A
needs no model at all.

**Rejected.** WSL plus vLLM, which is days of setup for a capability the first plan does not use.

**Cost if wrong.** Ollama is slower than vLLM for high-throughput batch inference. At evaluation
scale, with disk caching, that does not matter.

## D-010 — A hosted free-tier model judges answers, not a local model (2026-10-18)

**Context.** Answer scoring needs a model as judge, calibrated against roughly twenty manual labels.

**Decision.** Use a free tier of a hosted open model — Groq, OpenRouter or Google AI Studio — rather
than a local 7B.

**Why.** Judge quality directly determines whether the reported numbers mean anything. A 7B model
quantized to fit 6 GB of VRAM is a mediocre judge, and a noisy judge produces numbers that cannot
be defended. The model interface is isolated in one module, so the choice is a one-line change.

**Rejected.** A local model, which trades real measurement quality for offline operation that this
project does not need.

**Cost if wrong.** Free tiers have rate limits, so a full evaluation run is paced rather than
instant, and on-disk response caching becomes mandatory rather than convenient.

## D-011 — The Streamlit inspector is a diagnostic tool, not the product surface (2026-10-18)

**Context.** A user interface was requested, partly to make the system easy to see and understand.

**Decision.** Add a small Streamlit retrieval inspector to Plan A. The deployed product surface —
authentication, feedback loop, hosting — remains deferred until after Plan B.

**Why.** A UI over a system whose retrieval quality cannot yet be measured is decoration. An
inspector that shows query to retrieved chunks with scores is the tool actually used while tuning,
and it is about forty lines. It also makes retrieval legible without reading a report.

**Rejected.** Building the full product surface now.

**Cost if wrong.** Streamlit is a dependency added for a tool, not for the product. If it is
dropped later, one module and one Makefile target go with it.

## D-012 — Ground-truth labels come from reverse generation plus human verification (2026-10-18)

**Context.** Labelling a golden set by hand risks requiring domain expertise the author does not
have, which was the main threat to the project.

**Decision.** Separate the two labels. Retrieval ground truth is "which chunks contain the answer",
which is verifiable by reading a clause. Answer ground truth is graded by citation match plus a
calibrated judge. Candidate questions are generated by taking a chunk and asking what question it
answers, then verified by hand.

**Why.** Reading comprehension is not tax expertise. Every answer in legislation is a quotable
clause, so ground truth is checkable without being a chartered accountant. Labelling effort drops
from weeks to a few hours.

**Rejected.** Purely hand-written questions, which would have required domain study; and unverified
generated questions, which would have produced numbers nobody should believe.

**Cost if wrong.** Generated questions can be trivially easy or subtly wrong. Verification by
reading, a written annotation guide, and a measured label-stability check are the controls.

## D-013 — The repo opens private, and the work email is stripped from git history (2026-10-18)

**Context.** The repository was created during the first working session. The global git
configuration used a work email address (`@kirloskar.com`), which would have been embedded in every
commit.

**Decision.** Create the repository private, set the repository-local commit email to the GitHub
noreply address, and rewrite the existing commits before the first push.

**Why.** Commit emails are scraped. A work address on a personal side project is a small but real
leak, and rewriting four commits is trivial where rewriting a year of history is not. Private allows
the project to be half-built without being judged at its worst.

**Rejected.** Pushing straight to a public repository with the work email intact.

**Cost if wrong.** None if the repository later opens public, since history is already clean.

## D-014 — Archify receipt JSON files are not committed (2026-10-18)

**Context.** The diagram build emits `*.finalize.json`, `*.finalize-summary.json`,
`*.delivery.json` and `*.browser-check.json` alongside each diagram.

**Decision.** Commit the `.html` diagrams and their `.json` sources. Do not commit the receipts.

**Why.** The receipts are validation output, regenerable on demand, and they are tens of kilobytes
each of tool scaffolding in a portfolio repository.

**Rejected.** Committing everything, on the grounds that completeness is harmless. It is not: noise
in a repository makes the author look like they do not curate.

**Cost if wrong.** The build provenance of a committed diagram is not recorded. Regenerating the
diagram restores it.

## D-015 — Emptying the index drops the local storage directory (2026-10-18)

**Context.** `ChunkIndex.recreate` has to return an empty collection. The ingest CLI relies on it,
and it is the only way to rebuild the index after a chunking change.

**Decision.** `recreate` closes the client, removes the Qdrant storage directory, and opens a fresh
client.

**Why.** Measured on `qdrant-client` 1.19.1 in local mode: `delete_collection` followed by
`create_collection` reports the collection as gone, but the points are still there. `count()`
returns 1 and `retrieve` hands back the old point — including from a client freshly reopened on the
same path, so the deletion is not persisted to disk at all. The sequence the plan specifies silently
fails to empty the index. Removing the directory demonstrably works: `count()` returns 0 and the old
point is unreachable.

**Rejected.** `delete_collection` plus `create_collection` as specified, which is insufficient in
local mode; and deleting points by id before recreating, which is more code for the same outcome.

**Cost if wrong.** The directory is removed rather than emptied in place, so anything else stored
beside it would go too. Nothing else is: the directory holds only this collection.

## D-016 — The parser cuts the site's trailing chrome before chunking (2026-10-18)

**Context.** Every RBI notification page appends an archive tree and inline stylesheet fragments
after the signed text of the circular. Extracting the page's main container kept them, because they
sit inside the same container as the body.

**Decision.** `parse_document` truncates the body at the first chrome marker: a year followed by
`All Months`, `/*added by ni*/`, `DateSettree`, or `.accordion`.

**Why.** Measured on the captured corpus: 207 chunks before the cut, 154 after, with the same 25
documents and zero parse failures. About a quarter of the corpus was navigation markup, and it
competed directly with real provisions in vector search. It also inflated the chunk count into a
misleading measure of corpus size.

**Rejected.** Tightening the container selector, which cannot separate the tree from the body
because both live in one container; and leaving the markup in place, on the grounds that a retriever
should learn to ignore it. It should not have to, and the retrieved chunks were unreadable.

**Cost if wrong.** A genuine notification body containing one of those marker strings would be
truncated early and its tail lost. No RBI circular body uses them; `tests/test_rbi_parse.py` asserts
the cut on the captured fixture, and the parse-failure count on the full corpus is zero.

## D-017 — A chunk carries a document-level citation string (2026-10-18)

**Context.** The inspector and the candidate sheet both read `citation` from chunk metadata. Nothing
produced one: `parse_document` emitted `published` and `circular` only, so every citation rendered
empty.

**Decision.** `parse_document` adds `citation` to document metadata as `<circular> <title>`, which
chunks inherit.

**Why.** A retrieved chunk has to be citable to a human, and a golden-set label has to be readable as
a document rather than a bare id. The value is composed at parse time from fields already extracted,
so it costs nothing downstream.

**Rejected.** Composing the citation in the inspector, which would leave every other consumer
without one; and a provision-level citation, which the parser does not yet know how to extract.

**Cost if wrong.** The citation identifies the document, not the clause, so it does not tell a reader
which part of the circular a chunk came from. Provision-level citation is the upgrade path.

## D-018 — Unanswerable questions are aggregated, and the cap is stated (2026-10-18)

**Context.** Two of the twenty golden questions are unanswerable by design and have no relevant
chunk, so they score zero on every retrieval metric. The evaluation aggregates over all twenty.

**Decision.** Keep them in the aggregate for this milestone, and state plainly in `README.md` and
`data/golden/README.md` that recall cannot therefore exceed 0.90 on this set.

**Why.** The category mix is what the design asks for, and there is no abstention mechanism yet, so
there is nothing else to score an unanswerable question against. Reporting one aggregate with a
stated ceiling is honest; quietly dropping the questions would hide a real property of the set.

**Rejected.** Excluding unanswerable questions from the aggregate; and per-category reporting in this
milestone, which is more report surface than the baseline needs.

**Cost if wrong.** The headline figures understate retrieval quality on answerable questions by up to
0.10. A reader who misses the caveat reads the baseline as worse than it is. Per-category reporting
and abstention both arrive with answer generation.
