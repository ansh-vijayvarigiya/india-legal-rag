"""Interactive retrieval inspector. `python -m legalrag.inspector`.

A diagnostic tool, not a product surface: it exists so that retrieval quality can
be judged by eye while retrieval is being changed. The product surface is later
work, after answer generation exists.

`results_rows` is separated from the rendering so the table contents can be
tested without running Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence

from legalrag.config import Settings
from legalrag.corpus import read_chunks
from legalrag.embed import Embedder
from legalrag.index import ChunkIndex
from legalrag.models import Retrieved
from legalrag.retrieve import DenseRetriever

DEFAULT_K = 10
MAX_K = 50


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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Retrieval inspector", layout="wide")

    @st.cache_resource
    def load() -> tuple[DenseRetriever, int]:
        return build_retriever(Settings())

    try:
        retriever, chunk_count = load()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
        return

    st.sidebar.metric("chunks indexed", chunk_count)

    query = st.text_input("Query")
    k = st.number_input("Results", min_value=1, max_value=MAX_K, value=DEFAULT_K)

    if not query:
        st.caption("Type a query. Nothing is sent anywhere; retrieval runs locally.")
        return

    rows = results_rows(retriever.retrieve(query, k=int(k)))
    st.dataframe([{key: value for key, value in row.items() if key != "text"} for row in rows])
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
