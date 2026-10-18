from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings. Override any field with a LEGALRAG_ prefixed env var."""

    model_config = SettingsConfigDict(env_prefix="LEGALRAG_")

    data_dir: Path = Path("data")
    qdrant_path: Path = Path("data/qdrant")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    collection: str = "chunks-bge-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()
