from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings. Override any field with a LEGALRAG_ prefixed env var.

    `.env` is read when present, so the API key can live in one git-ignored file
    instead of a shell session. Environment variables win over the file.
    """

    model_config = SettingsConfigDict(env_prefix="LEGALRAG_", env_file=".env")

    data_dir: Path = Path("data")
    qdrant_path: Path = Path("data/qdrant")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    collection: str = "chunks-bge-small"

    # Answer generation. Empty key means the app runs without AI rather than
    # failing, so retrieval and its evaluation stay usable with no account.
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    # A different model grades the answers. One model marking its own work
    # shares its own blind spots.
    judge_model: str = "qwen/qwen3.8-27b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
