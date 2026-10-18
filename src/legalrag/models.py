from dataclasses import dataclass


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    """Stable identifier for a chunk. Golden-set labels reference these."""
    return f"{doc_id}#{ordinal:04d}"


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    source: str
    source_url: str
    title: str
    published: str | None
    text: str
    metadata: dict[str, str]

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "source_url": self.source_url,
            "title": self.title,
            "published": self.published,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Document":
        return cls(
            doc_id=obj["doc_id"],
            source=obj["source"],
            source_url=obj["source_url"],
            title=obj["title"],
            published=obj["published"],
            text=obj["text"],
            metadata=dict(obj["metadata"]),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    embed_text: str
    metadata: dict[str, str]

    def to_json(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "embed_text": self.embed_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Chunk":
        return cls(
            chunk_id=obj["chunk_id"],
            doc_id=obj["doc_id"],
            ordinal=obj["ordinal"],
            text=obj["text"],
            embed_text=obj["embed_text"],
            metadata=dict(obj["metadata"]),
        )


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: Chunk
    score: float
    rank: int
