"""Retrieval inspector and answer page. `python -m legalrag.inspector`.

Two views over the same index. "Answer" asks a model to read the retrieved
passages and reply in plain English, with the passages kept underneath so the
answer can be checked rather than trusted. "Passages" is the raw ranked list,
which is what you want when an answer looks wrong and the retrieval behind it
needs to be seen.

The pure parts — row building, example selection, error wording — are separated
from the rendering so they can be tested without running Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from legalrag.answer import LLMError, MAX_PASSAGES, OpenAIChatClient, answer
from legalrag.config import Settings
from legalrag.corpus import read_chunks
from legalrag.embed import Embedder
from legalrag.golden import GoldenSetError, load_golden_set
from legalrag.index import ChunkIndex
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
    text = str(error)
    if "429" in text or "rate limit" in text.lower():
        return (
            "The free AI plan allows only a few questions a minute, and that "
            "allowance is used up. Wait about twenty seconds and ask again. "
            "The passages below were already retrieved and cost nothing."
        )
    return text


def answer_client(settings: Settings) -> OpenAIChatClient | None:
    """The chat client, or None when no key is configured."""
    key = settings.llm_api_key.get_secret_value()
    if not key:
        return None
    return OpenAIChatClient(settings.llm_base_url, key, settings.llm_model)


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


def _render_answer(st, question: str, hits: Sequence[Retrieved], client) -> None:
    if client is None:
        st.warning(
            "No AI key configured, so there is no written answer here — the "
            "passages below are what retrieval found. To switch answers on, put "
            "`LEGALRAG_LLM_API_KEY=...` in a `.env` file in the project root."
        )
        return

    with st.spinner("Reading the passages..."):
        try:
            result = answer(question, hits, client)
        except LLMError as error:
            st.error(rate_limit_message(error))
            st.caption("Showing the retrieved passages instead.")
            return

    st.markdown(result.text)
    if result.used_chunk_ids:
        st.caption(f"Answered from {len(result.used_chunk_ids)} of the passages below.")
    else:
        st.caption("The model answered without citing a passage. Check it against them.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RBI regulatory search", layout="wide")

    @st.cache_resource
    def load() -> tuple[DenseRetriever, int]:
        return build_retriever(Settings())

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
        "Search runs locally; only a question and the passages it retrieves are "
        "sent to the language model."
    )

    answer_tab, passages_tab = st.tabs(["Answer", "Passages"])
    client = answer_client(settings)

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
            hits = retriever.retrieve(question, k=MAX_PASSAGES)
            _render_answer(st, question, hits, client)
            if hits:
                # With no key the passages are the whole answer, so show them.
                _render_sources(st, hits, expanded=client is None)

    with passages_tab:
        st.caption("The raw ranked list, for checking what retrieval actually returned.")
        query = st.text_input("Query", key="passages_query")
        k = st.number_input("Results", min_value=1, max_value=MAX_K, value=DEFAULT_K)

        if not query:
            st.caption("Type a query. Nothing is sent anywhere; retrieval runs locally.")
            return

        rows = results_rows(retriever.retrieve(query, k=int(k)))
        st.dataframe(
            [{key: value for key, value in row.items() if key != "text"} for row in rows]
        )
        for row in rows:
            with st.expander(f"{row['rank']}. {row['chunk_id']} - {row['citation']}"):
                st.write(row["text"])


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
