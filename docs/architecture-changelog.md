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
