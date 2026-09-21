"""Smoke test for the Streamlit app using Streamlit's headless AppTest, with a fake LLM and fake GitHub."""

from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_agent import FakeOpenAI, github, reply
from tools.client import GitHubClient, set_client

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # MemoryStore() writes agent_memory.db into the working directory
    st.cache_resource.clear()
    set_client(GitHubClient(transport=httpx.MockTransport(github), sleep=lambda _: None))
    llm = FakeOpenAI(
        reply("Checking metadata.", [("get_repository", {})]),
        reply("octo/demo is a small demo."),
        reply("Second answer."),
    )
    monkeypatch.setattr("agents.single.resolve_llm", lambda client=None, model=None: (llm, "gpt-4o-mini"))
    at = AppTest.from_file(APP, default_timeout=15)
    yield at, llm
    set_client(None)
    st.cache_resource.clear()


def test_investigation_flow_and_memory_resume(app):
    at, llm = app
    at.run()
    assert not at.exception and "Enter a public repository" in at.info[0].value

    at.sidebar.text_input(key="repo_text").set_value("https://github.com/octo/demo").run()
    at.chat_input[0].set_value("Analyze it").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "octo/demo is a small demo." in at.chat_message[1].markdown[-1].value
    assert any("get_repository" in e.label for e in at.expander)  # tool call card
    assert any("Saved repository profile" in c.value for c in at.caption)

    # a fresh browser session (new AppTest) resumes the stored conversation from SQLite
    again = AppTest.from_file(APP, default_timeout=15)
    again.run()
    again.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not again.exception
    assert [m.name for m in again.chat_message] == ["user", "assistant"]
    assert any("Used 1 tool call" in c.value for c in again.caption)


def test_invalid_repo_shows_error(app):
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("not a repo").run()
    assert not at.exception and "not a valid repository" in at.error[0].value


def test_multi_agent_mode_is_placeholder(app):
    at, _ = app
    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    assert not at.exception and "War Room" in at.info[0].value
