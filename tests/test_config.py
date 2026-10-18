from pathlib import Path

from legalrag.config import Settings, get_settings


def test_settings_have_expected_defaults():
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.qdrant_path == Path("data/qdrant")


def test_env_var_overrides_data_dir(monkeypatch):
    monkeypatch.setenv("LEGALRAG_DATA_DIR", "D:/tmp/legalrag")
    get_settings.cache_clear()
    assert get_settings().data_dir == Path("D:/tmp/legalrag")
    get_settings.cache_clear()
