# Architecture changelog

Append-only. One entry per change to the system's shape, as opposed to a change inside it. Rationale
lives in `decisions.md`; requirements live in the spec.

---

## 2026-10-18 — Initial architecture, retrieval-only

**Spec:** `superpowers/specs/2026-10-18-indian-regulatory-rag-design.md`
**Plan:** `superpowers/plans/2026-10-18-corpus-and-retrieval-baseline.md`
**Branch:** `plan-a-baseline`

**What exists.** A linear offline pipeline over RBI notification pages: fetch into raw HTML
snapshots, parse into `Document` records, split into fixed-size `Chunk`s, embed locally with
`bge-small-en-v1.5`, and index into Qdrant in local on-disk mode. A separate evaluation path loads a
hand-verified golden set, runs a retriever over it, computes retrieval metrics, and renders
`results.md`.

**Shape.** Two entry points over a shared core — `legalrag.ingest` writes artifacts, `legalrag.eval`
reads them — plus `legalrag.inspector` for interactive inspection. The `Retriever` protocol is the
seam that later retrieval configurations plug into.

**Deliberately absent.** The spec describes more than this build contains, and the diagrams mark the
difference: India Code and CBIC sources, structure-aware chunking, the metadata and temporal model,
hybrid retrieval, Reciprocal Rank Fusion, cross-encoder reranking, query transformation, answer
generation, citations, refusal, the language-model judge, the FastAPI service, trace logging, and any
deployed interface.

**Constraints that drove the shape.**

- No Docker on the development machine, which set the vector store (Qdrant local mode, D-004) and, by
  extension, the tracing approach (D-005).
- A 6 GB Blackwell GPU, which set local embeddings and reranker (D-006) and ruled out anything
  larger.
- The labelling risk, which set the two-kinds-of-label split (D-012).

**Known weak points at this shape.** Amendment-chain extraction is not attempted yet, so the
temporal behaviour the spec calls the project's strongest differentiator does not exist. Fixed-size
chunking is a deliberate placeholder and is the first ablation row. Retrieval is dense-only, so
lexical and hybrid comparisons are also unbuilt.

**Next shape change expected.** Plan B adds the answering layer: a model client, caching, grounded
answers with clause citations, refusal behaviour, and a calibrated judge for faithfulness and
correctness. That is when a model first enters the runtime.

---

## 2026-10-18 — Corpus contracted to real circular text

**Branch:** `plan-a-baseline`. Decisions D-016, D-017.

**What changed.** The parser now cuts the notification page's trailing archive tree and stylesheet
fragments, and stamps a document-level `citation` onto each chunk's metadata. The corpus went from
207 chunks to 154 chunks over the same 25 documents, none of them newly failing to parse.

**Why it is a shape change.** The first corpus was roughly a quarter navigation markup, and that
markup was indexed and retrieved on equal terms with provisions. Every retrieval number before this
change was measured against a corpus that was partly not a corpus, and the chunk count was not a
meaningful size measure. The cut shortened every multi-chunk document, which moved the chunk ids of
every chunk after the first one, so the corpus had to be re-ingested and the golden set relabelled
against the corrected corpus.

**Now checkable that was not before.** A retrieved result can be cited to a human as a document, so
the inspector shows something meaningful and a golden label can be read back as a document.

**Still weak.** The cut is marker-based rather than structural, and citations are document-level
rather than provision-level.
