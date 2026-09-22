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


def test_war_room_mode_asks_for_a_repository_first(app):
    at, _ = app
    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    assert not at.exception and "Enter a public repository" in at.info[0].value


def _war_room_button(at):
    return next(b for b in at.button if "Run War Room" in b.label)


def test_war_room_runs_the_team_and_shows_then_reloads_the_report(app, monkeypatch):
    from tests.test_war_room import team_script

    at, _ = app
    llm = FakeOpenAI(*team_script())
    monkeypatch.setattr("orchestration.runner.resolve_llm", lambda client=None, model=None: (llm, "m"))

    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not at.exception and any("War Room for **octo/demo**" in c.value for c in at.caption)
    _war_room_button(at).click().run()

    assert not at.exception
    assert any("War Room finished" in s.value for s in at.success)
    markdown = "\n".join(m.value for m in at.markdown)
    assert any(s.value == "Repository Health Report: octo/demo" for s in at.subheader)
    assert "A small, well organized project" in markdown and ":gray[`#1`]" in markdown  # narrative, citations styled
    assert {m.label: m.value for m in at.metric if "Specialists" in m.label or "Findings" in m.label} == {"Findings": "4", "Specialists done": "4/4"}
    assert markdown.count("✅ done") >= 4  # the status dashboard
    assert {e.label for e in at.expander} >= {"Architecture (1)", "Security (1)"}  # evidence, one expander per specialist

    # a brand-new browser session finds the saved report in SQLite, without re-running anything
    again = AppTest.from_file(APP, default_timeout=15)
    again.run()
    again.sidebar.radio[0].set_value("multi_agent").run()
    again.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not again.exception
    assert any("showing the last saved report" in c.value for c in again.caption)
    assert any("A small, well organized project" in m.value for m in again.markdown)
    assert {m.label: m.value for m in again.metric if m.label == "Specialists done"} == {"Specialists done": "4/4"}


def test_war_room_can_run_a_single_specialist(app, monkeypatch):
    from tests.test_war_room import NARRATIVE, save

    at, _ = app
    llm = FakeOpenAI(reply(None, [save("security_policy", "No SECURITY.md", "warning", 0.6)]), reply("Done."), reply(NARRATIVE))
    monkeypatch.setattr("orchestration.runner.resolve_llm", lambda client=None, model=None: (llm, "m"))
    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    at.multiselect[0].set_value(["security_specialist"]).run()
    _war_room_button(at).click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Findings"] == "1" and metrics["Specialists done"] == "1/1"
    assert "Security (1)" in {e.label for e in at.expander}


def test_war_room_shows_a_clear_error_when_the_llm_cannot_start(app, monkeypatch):
    def boom(client=None, model=None):
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr("orchestration.runner.resolve_llm", boom)
    at, _ = app
    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    _war_room_button(at).click().run()
    assert not at.exception and any("OPENAI_API_KEY" in e.value for e in at.error)
