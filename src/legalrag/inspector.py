"""Retrieval inspector, answer page, and the evaluation behind both.

Three views over the same index. "Answer" asks a model to read the retrieved
passages and reply in plain English, with the passages kept underneath so the
answer can be checked rather than trusted, and with a button that runs a second
model on that same answer to mark it. "Passages" is the raw ranked list, which is
what you want when an answer looks wrong and the retrieval behind it needs to be
seen. "Evaluation" is the scoreboard: retrieval is scored live against the 20
labelled questions, and the answer grading is read from the committed report.

The pure parts — row building, report splitting, example selection, error
wording — are separated from the rendering so they can be tested without running
Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from legalrag.answer import LLMError, MAX_PASSAGES, OpenAIChatClient, answer, is_rate_limit
from legalrag.config import Settings
from legalrag.corpus import read_chunks
from legalrag.embed import Embedder
from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.runner import EvalRun, run_eval
from legalrag.golden import GoldenQuestion, GoldenSetError, load_golden_set
from legalrag.index import ChunkIndex
from legalrag.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_REASONING,
    VerdictError,
    Verdict,
    judge_answer,
)
from legalrag.models import Retrieved
from legalrag.retrieve import DenseRetriever

DEFAULT_K = 10
MAX_K = 50
EXAMPLE_COUNT = 4


def results_rows(hits: Sequence[Retrieved]) -> list[dict]:
    """One display row per hit. Row order is the retrieval rank order."""
    return [
        {
            "rank": hit.rank,
            "score": round(hit.score, 4),
            "chunk_id": hit.chunk.chunk_id,
            "doc_id": hit.chunk.doc_id,
            "citation": hit.chunk.metadata.get("citation", ""),
            "text": hit.chunk.text,
            "chars": len(hit.chunk.text),
        }
        for hit in hits
    ]


def example_questions(golden_path: Path, n: int = EXAMPLE_COUNT) -> list[str]:
    """Answerable questions from the golden set.

    These are known to be answerable from this corpus, which matters more than
    variety: a visitor has no way to guess what this corpus covers, and a
    plausible-sounding question from outside it just returns a refusal.
    """
    try:
        questions = load_golden_set(golden_path)
    except (FileNotFoundError, GoldenSetError):
        return []
    return [question.question for question in questions if question.answerable][:n]


def rate_limit_message(error: LLMError) -> str:
    """Say what to do about a spent free-tier quota, rather than quoting it."""
    if is_rate_limit(error):
        return (
            "The free AI plan allows only a few questions a minute, and that "
            "allowance is used up. Wait about twenty seconds and ask again. "
            "The passages below were already retrieved and cost nothing."
        )
    return str(error)


def answer_client(settings: Settings) -> OpenAIChatClient | None:
    """The chat client, or None when no key is configured."""
    key = settings.llm_api_key.get_secret_value()
    if not key:
        return None
    return OpenAIChatClient(settings.llm_base_url, key, settings.llm_model)


def judge_client(settings: Settings) -> OpenAIChatClient | None:
    """The grading client, set up exactly as `legalrag.judge` sets it up.

    Same model, same reasoning setting and same reply budget as the batch run, so
    a mark given here is the mark the committed report would give.
    """
    key = settings.llm_api_key.get_secret_value()
    if not key:
        return None
    return OpenAIChatClient(
        settings.llm_base_url,
        key,
        settings.judge_model,
        max_tokens=JUDGE_MAX_TOKENS,
        reasoning_effort=JUDGE_REASONING,
    )


def metrics_rows(metrics: RetrievalMetrics) -> list[dict]:
    """The headline numbers, in the order `results.md` reports them."""
    return [
        {"metric": "recall@5", "score": round(metrics.recall[5], 2)},
        {"metric": "recall@10", "score": round(metrics.recall[10], 2)},
        {"metric": "recall@20", "score": round(metrics.recall[20], 2)},
        {"metric": "precision@5", "score": round(metrics.precision[5], 2)},
        {"metric": "mrr", "score": round(metrics.mrr, 2)},
        {"metric": "ndcg@10", "score": round(metrics.ndcg[10], 2)},
    ]


def question_rows(run: EvalRun, questions: Sequence[GoldenQuestion]) -> list[dict]:
    """One row per golden question, saying whether retrieval found the label.

    An unanswerable question has nothing to find, so it is reported as `n/a`
    rather than as a miss. Reporting it as a miss would show the correct result
    as a failure.
    """
    by_qid = {question.qid: question for question in questions}
    rows: list[dict] = []
    for result in run.results:
        question = by_qid.get(result.qid)
        relevant = set(result.relevant_ids)
        if not relevant:
            rank, found, labelled, in_top_5 = 0, "n/a", "none, by design", "n/a"
        else:
            rank = next(
                (
                    position
                    for position, chunk_id in enumerate(result.retrieved_ids, start=1)
                    if chunk_id in relevant
                ),
                0,
            )
            found = "yes" if rank else "no"
            labelled = ", ".join(sorted(relevant))
            # Counted, not just ranked: a question whose second labelled chunk
            # sits at rank 15 is a partial hit, and "found: yes" would hide it.
            in_top_5 = f"{len(set(result.retrieved_ids[:5]) & relevant)} of {len(relevant)}"
        rows.append(
            {
                "qid": result.qid,
                "type": question.category if question else "",
                "labelled chunks": labelled,
                "labelled in top 5": in_top_5,
                "first labelled at rank": rank,
                "found in top 20": found,
            }
        )
    return rows


def answers_report(markdown: str) -> tuple[str, str]:
    """Split the committed grading report into its summary and its detail.

    The summary is what a visitor needs — the counts and the verdict table. The
    per-answer sections are long, and they belong behind a fold.
    """
    summary, separator, detail = markdown.partition("\n### ")
    if not separator:
        return summary.strip(), ""
    return summary.strip(), f"### {detail}"


def build_retriever(settings: Settings) -> tuple[DenseRetriever, int]:
    """Rebuild the index from the chunk store and return it with the chunk count."""
    chunks = list(read_chunks(settings.data_dir / "chunks" / "chunks.jsonl"))
    if not chunks:
        raise FileNotFoundError(
            f"no chunks at {settings.data_dir / 'chunks' / 'chunks.jsonl'}, "
            "run `python -m legalrag.ingest` first"
        )

    embedder = Embedder(settings.embedding_model)
    index = ChunkIndex(settings.qdrant_path, settings.collection, embedder.dim)
    # The chunk store is the source of truth here, so the index is rebuilt rather
    # than trusted to match it. The client is left open: Streamlit owns the
    # process for as long as the page is up.
    index.recreate()
    index.upsert(chunks, embedder.encode([chunk.embed_text for chunk in chunks]))
    return DenseRetriever(index, embedder), len(chunks)


def _source_heading(position: int, hit: Retrieved) -> str:
    citation = hit.chunk.metadata.get("citation", "")
    return f"**{position}.** `{hit.chunk.chunk_id}`" + (f" · {citation}" if citation else "")


def _render_sources(st, hits: Sequence[Retrieved], *, expanded: bool) -> None:
    """Numbered to match the `[1]` markers in the answer."""
    with st.expander(f"Where this came from — {len(hits)} passage(s)", expanded=expanded):
        for position, hit in enumerate(hits, start=1):
            st.markdown(_source_heading(position, hit))
            st.write(hit.chunk.text)
            if position < len(hits):
                st.divider()


def _answer_entry(st, question: str, retriever: DenseRetriever, client) -> dict:
    """Retrieve and answer once per question, not once per interaction.

    Streamlit re-runs the whole script on every click. Without this, pressing the
    grade button would ask the model for a fresh answer and spend a second call
    doing it, then grade a different answer than the one on the screen.
    """
    cached = st.session_state.get("answer")
    if cached is not None and cached["question"] == question:
        return cached

    hits: list[Retrieved] = retriever.retrieve(question, k=MAX_PASSAGES)
    entry: dict = {
        "question": question,
        "hits": hits,
        "result": None,
        "error": None,
        "verdict": None,
    }
    if client is not None:
        with st.spinner("Reading the passages..."):
            try:
                entry["result"] = answer(question, hits, client)
            except LLMError as error:
                entry["error"] = error
    st.session_state["answer"] = entry
    return entry


def _render_answer(st, entry: dict, client) -> None:
    if client is None:
        st.warning(
            "No AI key configured, so there is no written answer here — the "
            "passages below are what retrieval found. To switch answers on, put "
            "`LEGALRAG_LLM_API_KEY=...` in a `.env` file in the project root."
        )
        return

    if entry["error"] is not None:
        st.error(rate_limit_message(entry["error"]))
        st.caption("Showing the retrieved passages instead.")
        return

    result = entry["result"]
    st.markdown(result.text)
    if result.used_chunk_ids:
        st.caption(f"Answered from {len(result.used_chunk_ids)} of the passages below.")
    else:
        st.caption("The model answered without citing a passage. Check it against them.")


def _render_grading(st, entry: dict, judge) -> None:
    """The second model's marks on the answer above, on request.

    Grading is a button rather than automatic because it is a second paid call,
    and on a free plan a second call is a third of a minute's allowance.
    """
    if judge is None or entry["result"] is None:
        return

    verdict: Verdict | None = entry["verdict"]
    if verdict is None:
        st.caption(
            "A second, different model can mark this answer against the passages "
            "it was written from. That is one more model call."
        )
        if not st.button("Grade this answer"):
            return
        with st.spinner("Grading the answer..."):
            try:
                verdict = judge_answer(
                    entry["question"], entry["result"].text, entry["hits"], judge
                )
            except LLMError as error:
                st.error(rate_limit_message(error))
                return
            except VerdictError as error:
                st.error(f"The grader's reply could not be read: {error}")
                return
        entry["verdict"] = verdict

    left, middle, right = st.columns(3)
    left.metric("Answers the question", verdict.correct)
    middle.metric("Backed by the passages", verdict.grounded)
    right.metric("Refused", "yes" if verdict.refused else "no")
    st.caption(f"Why: {verdict.reason}")
    st.caption(
        "A model marking a model is a second opinion, not proof. Every reason is "
        "checkable against the passages below."
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RBI regulatory search", layout="wide")

    @st.cache_resource
    def load() -> tuple[DenseRetriever, int]:
        return build_retriever(Settings())

    @st.cache_data(show_spinner=False)
    def load_eval(_retriever, chunk_count: int, golden_path: str):
        """Cached: twenty retrievals should not re-run on every widget click.

        Keyed on the chunk count as well, so re-ingesting a different corpus
        invalidates the scoreboard instead of leaving a stale one on the page.
        """
        try:
            questions = load_golden_set(Path(golden_path))
        except (FileNotFoundError, GoldenSetError):
            return None, []
        if not questions:
            return None, []
        return run_eval(_retriever, questions, config="dense-fixed"), questions

    settings = Settings()
    try:
        retriever, chunk_count = load()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
        return

    st.title("RBI regulatory search")
    st.caption(
        f"{chunk_count} passages from the 25 RBI circulars in this corpus. "
        "Search runs locally. Retrieval is scored live in the Evaluation tab; "
        "only a question and the passages it retrieves are sent to a model."
    )

    answer_tab, passages_tab, evaluation_tab = st.tabs(["Answer", "Passages", "Evaluation"])
    client = answer_client(settings)
    judge = judge_client(settings)

    with answer_tab:
        question = st.text_input("Ask a question", key="answer_question")

        examples = example_questions(settings.data_dir / "golden" / "milestone1.yaml")
        if examples:
            st.caption("Or click one of the questions this corpus was tested with:")
            for column, example in zip(st.columns(len(examples)), examples):
                if column.button(example, use_container_width=True):
                    question = example

        if not question:
            st.info("Type a question, or click one of the test questions above.")
        else:
            entry = _answer_entry(st, question, retriever, client)
            _render_answer(st, entry, client)
            _render_grading(st, entry, judge)
            if entry["hits"]:
                # With no key the passages are the whole answer, so show them.
                _render_sources(st, entry["hits"], expanded=client is None)

    with passages_tab:
        st.caption("The raw ranked list, for checking what retrieval actually returned.")
        query = st.text_input("Query", key="passages_query")
        k = st.number_input("Results", min_value=1, max_value=MAX_K, value=DEFAULT_K)

        if not query:
            st.caption("Type a query. Nothing is sent anywhere; retrieval runs locally.")
        else:
            rows = results_rows(retriever.retrieve(query, k=int(k)))
            st.dataframe(
                [{key: value for key, value in row.items() if key != "text"} for row in rows]
            )
            for row in rows:
                with st.expander(f"{row['rank']}. {row['chunk_id']} - {row['citation']}"):
                    st.write(row["text"])

    with evaluation_tab:
        st.caption(
            "Retrieval is scored here, live, against the 20 labelled questions this "
            "project is built around. The answer grading is read from the committed "
            "report, because re-running it costs forty model calls."
        )

        golden_path = settings.data_dir / "golden" / "milestone1.yaml"
        run, questions = load_eval(retriever, chunk_count, str(golden_path))
        if run is None:
            st.info(f"No golden set at {golden_path}, so there is nothing to score against.")
        else:
            st.subheader("Retrieval")
            st.dataframe(metrics_rows(run.metrics), hide_index=True)
            st.caption(
                "Per question. A rank of 0 means retrieval returned no labelled "
                "passage at any depth. Unanswerable questions have no labelled "
                "passage, so they cannot be a miss."
            )
            st.dataframe(question_rows(run, questions), hide_index=True)

        st.subheader("Answers")
        answers_path = settings.data_dir.parent / "answers.md"
        try:
            markdown = answers_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            st.info(f"No {answers_path.name} yet. Run `make judge` to write one.")
        else:
            summary, detail = answers_report(markdown)
            st.markdown(summary)
            if detail:
                with st.expander("Every answer, and the judge's reason for each mark"):
                    st.markdown(detail)


if __name__ == "__main__":
    import streamlit.runtime as st_runtime

    if st_runtime.exists():
        main()
    else:
        # `streamlit run` executes this file with the runtime up; a bare
        # `python -m legalrag.inspector` does not, and would only warn about
        # missing script context. Hand over to the server instead.
        import sys

        import streamlit.web.cli as stcli

        sys.argv = [
            "streamlit",
            "run",
            # Headless skips the first-run "enter your email" prompt, which
            # otherwise blocks the server forever on any machine that has not
            # run Streamlit before, with no output explaining why.
            "--server.headless=true",
            *sys.argv[1:],
            __file__,
        ]
        raise SystemExit(stcli.main())
