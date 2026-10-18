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


def test_llm_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LEGALRAG_LLM_API_KEY", "gsk_from_env")
    monkeypatch.setenv("LEGALRAG_LLM_MODEL", "some/model")
    monkeypatch.setenv("LEGALRAG_JUDGE_MODEL", "other/judge")
    settings = Settings()
    assert settings.llm_api_key.get_secret_value() == "gsk_from_env"
    assert settings.llm_model == "some/model"
    assert settings.judge_model == "other/judge"


def test_llm_settings_have_usable_defaults():
    # _env_file=None keeps a developer's local .env out of the assertion.
    settings = Settings(_env_file=None)
    assert settings.llm_base_url.startswith("https://")
    assert settings.llm_model
    assert settings.judge_model
    assert settings.judge_model != settings.llm_model
    assert settings.llm_api_key.get_secret_value() == ""


def test_the_api_key_cannot_be_printed_by_accident(monkeypatch):
    monkeypatch.setenv("LEGALRAG_LLM_API_KEY", "gsk_must_not_leak")
    assert "gsk_must_not_leak" not in repr(Settings())
    assert "gsk_must_not_leak" not in str(Settings())
