"""Tests for Config (memory backend, secrets bridging) and session-wide usage tracking."""

import os

import pytest
import streamlit as st

from agents.usage import UsageMeter
from config import Config


def test_memory_backend_defaults_to_sqlite():
    assert Config(openai_api_key="k").memory_backend == "sqlite"


def test_memory_backend_accepts_memory():
    assert Config(openai_api_key="k", memory_backend="memory").memory_backend == "memory"


def test_memory_backend_rejects_unknown_values():
    with pytest.raises(ValueError):
        Config(openai_api_key="k", memory_backend="postgres")


def test_usage_meter_merge_folds_totals_without_mutating_the_source():
    total = UsageMeter(calls=1, prompt_tokens=100, cached_tokens=10, completion_tokens=20)
    other = UsageMeter(calls=2, prompt_tokens=50, cached_tokens=5, completion_tokens=10)
    total.merge(other)
    assert total == UsageMeter(calls=3, prompt_tokens=150, cached_tokens=15, completion_tokens=30)
    assert other == UsageMeter(calls=2, prompt_tokens=50, cached_tokens=5, completion_tokens=10)


# -- app.py: store backend selection and secrets bridging --------------------------------

def test_get_store_memory_backend_does_not_touch_disk(tmp_path, monkeypatch):
    import app

    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()
    store = app.get_store("memory")
    try:
        assert store.db_path == ":memory:"
        assert not (tmp_path / "agent_memory.db").exists()
    finally:
        store.close()
        st.cache_resource.clear()


def test_get_store_sqlite_backend_writes_the_expected_file(tmp_path, monkeypatch):
    import app

    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()
    store = app.get_store("sqlite")
    try:
        assert store.db_path == "agent_memory.db"
        assert (tmp_path / "agent_memory.db").exists()
    finally:
        store.close()
        st.cache_resource.clear()


def test_load_secrets_into_env_bridges_missing_vars_without_overriding(monkeypatch):
    import app

    monkeypatch.delenv("TEST_BRIDGED_KEY", raising=False)
    monkeypatch.setenv("TEST_ALREADY_SET_KEY", "original")
    monkeypatch.setattr(app.st, "secrets", {"TEST_BRIDGED_KEY": "from-secrets", "TEST_ALREADY_SET_KEY": "should-not-win"})

    app._load_secrets_into_env()

    assert os.environ["TEST_BRIDGED_KEY"] == "from-secrets"
    assert os.environ["TEST_ALREADY_SET_KEY"] == "original"


def test_load_secrets_into_env_is_a_noop_when_no_secrets_are_configured(monkeypatch):
    import app

    class ExplodingSecrets:
        def items(self):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(app.st, "secrets", ExplodingSecrets())
    app._load_secrets_into_env()  # must not raise
