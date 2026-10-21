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

---

## 2026-10-18 — A model enters the runtime: answers, and grading of those answers

**Branch:** `main`. Decisions D-020, D-021, D-022.

**What changed.** Two new modules and one new dependency class. `legalrag.answer` sends a question
and the top five retrieved passages to a hosted model and returns prose carrying citation markers
that resolve back to chunk ids. `legalrag.judge` asks a *different* hosted model to mark every answer
on two independent axes and writes `answers.md` beside `results.md`. The inspector grew a second tab,
and `Settings` gained four fields for the endpoint, the two models and the key.

**Why it is a shape change.** This is the first time the system reaches a network at runtime, the
first time it produces something a reader consumes as prose instead of as a ranked list, and the first
time an evaluation measures anything other than retrieval. It also introduces a trust boundary the
previous shape did not have: the corpus used to be the only source of text, and now a model authors
text that a reader may believe.

**What did not move.** The retrieval path and its metrics are untouched, because the model is not in
it. `results.md` still reproduces byte-identically, and the answer layer is additive to the harness
rather than a rewrite of it. The API key did not move into the repository either: it lives in a
git-ignored `.env`, settings hold it as a `SecretStr`, and a test asserts that a key cannot appear in
an error message.

**Now checkable that was not before.** A generated answer carries a number and a reason, so a
regression in answer quality is visible in a diff the same way a retrieval regression is. Previously
the only way to know an answer had got worse was to read one.

**Still weak.** Answers cost time and money, and a free plan permits roughly three questions a minute.
The grader is itself a model, so answer quality is an opinion with a reason attached rather than a
measurement, and it inherits some of the writer's blind spots. Refusal is a prompt instruction rather
than a calibrated confidence threshold, so it is a behaviour that can be observed but not tuned.

## 2026-10-21 - The evaluation moves into the interface

**What changed.** `legalrag.inspector` gained a third tab, `Evaluation`, and the Answer tab gained a
`Grade this answer` button. The tab scores retrieval live over the golden set and renders the committed
answer-grading summary from `answers.md`; the button runs `legalrag.judge` on the answer currently on
the screen. Answers are now cached per question in the Streamlit session, so a widget click no longer
buys a fresh answer.

**Shape.** No new entry point and no new dependency. The interface now reads both evaluation artifacts
- `results.md`'s numbers, recomputed, and `answers.md`, parsed into a summary and a detail - and calls
the same `judge_answer` the batch runner calls, with the same client settings. The row builders and the
report splitter are pure functions next to the existing ones, which is why the tab has tests without a
Streamlit runtime.

**Deliberately absent.** No blended quality score, no automatic grading on every answer, no charts, and
no second bespoke UI: the tab is `st.dataframe` and `st.markdown` over the artifacts that already exist.

**Now checkable that was not before.** Whether the answer in front of a reader is right and backed up is
visible beside it, and a partial retrieval (`q012`, `q013`, `q016`) is visible as `1 of 2` or `2 of 4`
instead of a tick. A corpus that drifts from the committed baseline now shows up as the page disagreeing
with `results.md`.

**Still weak.** The live judge shares the free tier's output meter, so grading a second answer within a
minute can be refused; the page says to wait. The committed answer grading is a snapshot of one run, so
it can and will age. Retrieval is still the bottleneck the scoreboard makes obvious: `q008` is missed at
every depth.

## 2026-10-21 - The page becomes one page, written for a reader who is not an engineer

**What changed.** `legalrag.inspector` lost its tab bar. It is now a single page: question and answer,
then the scoreboard as a sentence and two tables, then the committed answer grading, then a collapsed
engineering section. The metric table gained a column that says what each number means, and the
per-question table shows the question text and a plain outcome instead of chunk ids.

**Shape.** Nothing moved between modules and no measurement changed - the same golden set, the same
metrics, the same committed report. What changed is presentation: section headings replace the tab bar,
and both row builders now return reader-facing rows instead of debug rows.

**Deliberately absent.** No explanation page in front of the tabs, no charts, no second "simple mode"
alongside the detailed one.

**Now checkable that was not before.** That a reader who does not work in this field can tell, without
being told, what the search is scored against and whether a question was found - which is the only
audience a public demo has.

**Still weak.** The per-question question text is shortened at 78 characters, so a long question is
readable but incomplete on the page. `precision@5` stays at 0.22 and needed a paragraph of explanation
rather than a better number; that is honest, but it is still a number a visitor must be talked out of
misreading.
