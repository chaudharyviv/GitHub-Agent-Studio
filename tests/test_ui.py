"""Tests for the Phase 6 UI: findings browser, report viewer, error boundary, repository card, sidebar flows."""

import time

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from agents.multi.report import AgentStatus
from memory import Finding, MemoryStore
from tests.test_agent import FakeOpenAI, reply
from tests.test_app import APP, app  # noqa: F401  (the `app` fixture: tmp cwd, fake GitHub, fake single-agent LLM)
from tests.test_war_room import NARRATIVE, team_script
from tools.client import GitHubClient, RateLimit, set_client
from ui.components import filter_findings, findings_csv
from ui.styles import format_markdown, severity_color


def finding(id, agent="security_specialist", severity="info", category="ci_security", text="text", evidence="", confidence=0.9):
    return Finding(id=id, session_id="s", agent=agent, severity=severity, category=category, finding=text, confidence=confidence,
                   evidence_json='{"evidence": "%s"}' % evidence if evidence else "{}")


SAMPLE = [
    finding(1, "architecture_specialist", "info", "structure", "Clean src layout", "src/click", 0.9),
    finding(2, "security_specialist", "warning", "ci_security", "Workflow lacks permissions", "pre-commit.yaml", 0.6),
    finding(3, "security_specialist", "critical", "secrets", "Key committed", ".env in tree", 0.9),
    finding(4, "health_specialist", "info", "activity", "Active project", "commits", 0.4),
]


# -- pure helpers -------------------------------------------------------------------------

def test_filter_findings_combines_every_filter():
    ids = lambda **kw: [f.id for f in filter_findings(SAMPLE, **kw)]  # noqa: E731
    assert ids() == [1, 2, 3, 4]
    assert ids(agents=["security_specialist"]) == [2, 3]
    assert ids(severities=["critical", "warning"]) == [2, 3]
    assert ids(categories=["structure", "activity"]) == [1, 4]
    assert ids(min_confidence=0.6) == [1, 2, 3]
    assert ids(search="KEY") == [3]  # case-insensitive, matches the finding text
    assert ids(search="pre-commit") == [2]  # ...and the evidence
    assert ids(agents=["security_specialist"], min_confidence=0.9) == [3]
    assert ids(search="nothing matches this") == []


def test_findings_csv_has_a_header_and_one_row_per_finding():
    lines = findings_csv(SAMPLE).strip().splitlines()
    assert lines[0] == "id,agent,severity,category,confidence,finding,evidence,created_at" and len(lines) == 5
    assert lines[3].split(",")[:4] == ["3", "security_specialist", "critical", "secrets"]


def test_severity_colors_and_citation_formatting():
    assert [severity_color(s) for s in ("critical", "warning", "info", "???")] == ["red", "orange", "blue", "gray"]
    assert format_markdown("Bad [#12][#15] but issue #304 is plain.") == "Bad :gray[`#12`]:gray[`#15`] but issue #304 is plain."


# -- findings browser ------------------------------------------------------------------------

def _inspector_script(findings):
    from ui.components import findings_inspector

    findings_inspector(findings)


def test_findings_browser_filters_and_reports_the_count():
    at = AppTest.from_function(_inspector_script, args=(SAMPLE,), default_timeout=15).run()
    assert not at.exception and any("4 of 4 findings" in c.value for c in at.caption)

    at.multiselect(key="findings_severity").set_value(["critical"]).run()
    assert any("1 of 4 findings" in c.value for c in at.caption)
    assert any("Key committed" in m.value for m in at.markdown) and not any("Clean src layout" in m.value for m in at.markdown)

    at.multiselect(key="findings_severity").set_value([]).run()
    at.text_input(key="findings_search").set_value("permissions").run()
    assert any("1 of 4 findings" in c.value for c in at.caption)

    at.text_input(key="findings_search").set_value("zzz").run()
    assert any("No findings match these filters." in i.value for i in at.info)


def test_findings_browser_with_no_findings_says_so():
    at = AppTest.from_function(_inspector_script, args=([],), default_timeout=15).run()
    assert not at.exception and any("No findings saved" in i.value for i in at.info)


# -- report viewer -----------------------------------------------------------------------------

def _viewer_script(markdown, narrative, statuses, findings):
    from ui.components import report_viewer

    report_viewer(markdown, narrative=narrative, statuses=statuses, findings=findings)


STATUSES = [
    AgentStatus(agent_id="architecture_specialist", title="Architecture", state="done", findings=1),
    AgentStatus(agent_id="security_specialist", title="Security", state="done", findings=2),
    AgentStatus(agent_id="health_specialist", title="Health", state="error", message="Could not reach OpenAI."),
]


def test_structured_report_has_metrics_warnings_narrative_and_evidence():
    at = AppTest.from_function(_viewer_script, args=("# md", "## Executive summary\nRisky [#3] and imaginary [#99].", STATUSES, SAMPLE[:3]), default_timeout=15).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Findings"] == "3" and metrics["🔴 Critical"] == "1" and metrics["🟠 Warning"] == "1" and metrics["Specialists done"] == "2/3"
    assert any("Health" in w.value and "Could not reach OpenAI." in w.value for w in at.warning)  # a failed specialist is called out
    assert any("do not exist: #99" in w.value for w in at.warning)  # a bad citation is called out
    assert any(":gray[`#3`]" in m.value for m in at.markdown)
    assert {e.label for e in at.expander} == {"Architecture (1)", "Security (2)"}  # no expander for the specialist with no findings


def test_report_viewer_falls_back_to_markdown_for_older_reports():
    at = AppTest.from_function(_viewer_script, args=("# Old report\nBody [#1]", None, None, None), default_timeout=15).run()
    assert not at.exception and any("Old report" in m.value and ":gray[`#1`]" in m.value for m in at.markdown)
    assert not at.metric


# -- tool timeline, rate limit caption -----------------------------------------------------------

def _timeline_script():
    from agents.base import AgentEvent
    from ui.components import tool_timeline

    tool_timeline([
        AgentEvent(kind="reasoning", step=1, content="thinking"),
        AgentEvent(kind="tool_result", step=1, name="get_repository", arguments={}, content="ok (468 chars)"),
        AgentEvent(kind="tool_result", step=2, name="get_file_content", arguments={"path": "x"}, content="not_found: nope", is_error=True),
    ])


def test_tool_timeline_lists_only_tool_calls():
    at = AppTest.from_function(_timeline_script, default_timeout=15).run()
    frame = at.dataframe[0].value
    assert list(frame["Tool"]) == ["get_repository", "get_file_content"] and list(frame["OK"]) == ["✅", "❌"]


def _rate_script():
    from ui.components import github_rate_limit_caption

    github_rate_limit_caption()


def test_rate_limit_caption_warns_when_the_budget_is_nearly_gone():
    client = GitHubClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    set_client(client)
    try:
        assert not AppTest.from_function(_rate_script, default_timeout=15).run().caption  # nothing seen yet: say nothing

        client._limits["core"] = RateLimit(remaining=4_000, reset_at=time.time() + 600)
        at = AppTest.from_function(_rate_script, default_timeout=15).run()
        assert "4,000 requests left" in at.caption[0].value and not at.warning

        client._limits["core"] = RateLimit(remaining=3, reset_at=time.time() + 600)
        at = AppTest.from_function(_rate_script, default_timeout=15).run()
        assert any("GITHUB_TOKEN" in w.value for w in at.warning)
    finally:
        set_client(None)


# -- error boundary -----------------------------------------------------------------------------------

def _boom_script():
    import streamlit as st

    from ui.components import safe_render

    def boom():
        raise RuntimeError("kaput")

    safe_render(boom)
    st.write("still running")


def _stop_script():
    import streamlit as st

    from ui.components import safe_render

    def stopper():
        st.write("before")
        st.stop()

    safe_render(stopper)
    st.write("after")


def test_a_failing_screen_becomes_a_friendly_message_not_a_traceback():
    at = AppTest.from_function(_boom_script, default_timeout=15).run()
    assert not at.exception  # the app did not crash
    assert "Something went wrong (RuntimeError)" in at.error[0].value and "memory is safe" in at.error[0].value
    assert "kaput" in at.code[0].value  # details are available on demand
    assert any("still running" in m.value for m in at.markdown)


def test_the_error_boundary_does_not_swallow_st_stop():
    at = AppTest.from_function(_stop_script, default_timeout=15).run()
    assert not at.exception and not at.error
    assert any("before" in m.value for m in at.markdown) and not any("after" in m.value for m in at.markdown)


def test_the_whole_app_survives_a_crashing_screen(app, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("screen exploded")

    monkeypatch.setattr("ui.single_agent.render_single_agent", crash)
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not at.exception and any("Something went wrong (RuntimeError)" in e.value for e in at.error)
    assert at.sidebar.text_input(key="repo_text").value == "octo/demo"  # the sidebar still works


# -- repository card and its failure states -----------------------------------------------------------------

def test_repository_card_shows_key_facts(app):
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Stars"] == "7" and metrics["Language"] == "Python" and metrics["License"] == "—"
    assert any(m.value == "### octo/demo" for m in at.markdown)


def test_unknown_repository_stops_before_any_agent_screen(app):
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/missing").run()
    assert not at.exception and any("octo/missing" in e.value and "was not found" in e.value for e in at.error)
    assert not at.tabs and not at.chat_input  # no chat, no War Room


def test_rate_limited_lookup_warns_but_lets_you_continue(app):
    def limited(request):
        return httpx.Response(403, json={"message": "API rate limit exceeded"},
                              headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(time.time()) + 300)})

    set_client(GitHubClient(transport=httpx.MockTransport(limited), sleep=lambda _: None))
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert not at.exception and any("rate limit" in w.value.lower() for w in at.warning)
    assert at.chat_input  # still usable: cached data or a token may get you through


# -- sidebar: mode switch keeps the repository -----------------------------------------------------------------

def test_switching_mode_keeps_the_repository_and_changes_the_screen(app):
    at, _ = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    assert at.chat_input and any("Investigating **octo/demo**" in c.value for c in at.caption)
    assert any("Analyze this repository" in b.label for b in at.sidebar.button)  # quick actions belong to the single agent

    at.sidebar.radio[0].set_value("multi_agent").run()
    assert at.sidebar.text_input(key="repo_text").value == "octo/demo"  # the repository survived the switch
    assert any("War Room for **octo/demo**" in c.value for c in at.caption) and not at.chat_input
    assert not any("Analyze this repository" in b.label for b in at.sidebar.button)

    at.sidebar.radio[0].set_value("single_agent").run()
    assert at.chat_input and at.sidebar.text_input(key="repo_text").value == "octo/demo"


# -- clearing memory needs a confirmation -----------------------------------------------------------------------

def _button(at, text):
    return next(b for b in at.sidebar.button if text in b.label)


def test_clear_memory_asks_first_and_cancel_keeps_everything(app):
    at, llm = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    at.chat_input[0].set_value("Analyze it").run()
    store = MemoryStore("agent_memory.db")
    assert store.get_repository_profile("octo", "demo") is not None

    _button(at, "Clear repository memory").click().run()
    assert any("Permanently delete" in w.value and "octo/demo" in w.value for w in at.sidebar.warning)
    assert store.get_repository_profile("octo", "demo") is not None  # asking deletes nothing

    at.sidebar.button(key="confirm_clear_no").click().run()
    assert not at.sidebar.warning
    assert store.get_repository_profile("octo", "demo") is not None and [m.name for m in at.chat_message] == ["user", "assistant"]
    store.close()


def test_confirmed_clear_deletes_memory_and_resets_the_screen(app):
    at, llm = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    at.chat_input[0].set_value("Analyze it").run()

    _button(at, "Clear repository memory").click().run()
    at.sidebar.button(key="confirm_clear_yes").click().run()
    assert not at.exception and not at.sidebar.warning
    store = MemoryStore("agent_memory.db")
    assert store.list_repositories() == [] and store.get_conversation_history("nothing") == []
    assert not at.chat_message  # the chat on screen is gone too
    store.close()


def test_clearing_memory_also_clears_a_saved_war_room_report(app, monkeypatch):
    at, _ = app
    llm = FakeOpenAI(*team_script())
    monkeypatch.setattr("orchestration.runner.resolve_llm", lambda client=None, model=None: (llm, "m"))
    at.run()
    at.sidebar.radio[0].set_value("multi_agent").run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    next(b for b in at.button if "Run War Room" in b.label).click().run()
    assert any("A small, well organized project" in m.value for m in at.markdown)

    _button(at, "Clear repository memory").click().run()
    at.sidebar.button(key="confirm_clear_yes").click().run()
    assert not at.exception
    assert any("No report yet" in i.value for i in at.info)  # the stale report is not still on screen
    assert not any("A small, well organized project" in m.value for m in at.markdown)


# -- what the user sees after a chat turn ----------------------------------------------------------------------------

def test_replayed_turns_show_a_tool_timeline_and_the_memory_tab_lists_sessions(app):
    at, llm = app
    at.run()
    at.sidebar.text_input(key="repo_text").set_value("octo/demo").run()
    at.chat_input[0].set_value("Analyze it").run()
    at.run()  # a rerun replays the stored turn instead of streaming it
    frames = [d.value for d in at.dataframe]
    assert any("Tool" in f.columns and "get_repository" in list(f["Tool"]) for f in frames)  # timeline in the steps expander
    assert any("Mode" in f.columns and "Single agent" in list(f["Mode"]) for f in frames)  # sessions table in Memory


# -- model text is escaped before it is shown as Markdown (found in a real browser check) ----------------

def test_escape_markdown_protects_file_names_and_leaves_code_spans_alone():
    from agents.multi.report import escape_markdown

    assert escape_markdown("Exported from src/click/__init__.py") == "Exported from src/click/\\_\\_init\\_\\_.py"
    assert escape_markdown("see `__init__.py` and a_b") == "see `__init__.py` and a\\_b"  # inside backticks: untouched
    assert escape_markdown("2 * 3 ~ 4 <tag> [x]") == "2 \\* 3 \\~ 4 \\<tag> \\[x\\]"
    assert escape_markdown("**bold** and __init__", keep_emphasis=True) == "**bold** and \\_\\_init\\_\\_"  # narrative keeps emphasis
    assert escape_markdown("a lone ` backtick and a_b") == "a lone ` backtick and a\\_b"


def test_demote_headings_moves_sections_below_the_report_title():
    from ui.styles import demote_headings

    assert demote_headings("## Executive summary\ntext\n### Sub") == "#### Executive summary\ntext\n##### Sub"
    assert demote_headings("###### deep") == "###### deep"  # never past h6
    assert demote_headings("not a #heading and ## not at start") == "not a #heading and ## not at start"


def test_downloadable_report_escapes_finding_text_too():
    from agents.multi.report import render_appendix

    f = Finding(id=1, session_id="s", agent="a_specialist", severity="info", category="c", finding="API in src/click/__init__.py",
                evidence_json='{"evidence": "src/click/__init__.py"}')
    appendix = render_appendix([f], [AgentStatus(agent_id="a_specialist", title="A")])
    assert "src/click/\\_\\_init\\_\\_.py" in appendix and "__init__" not in appendix


def _escaped_script(findings):
    from ui.components import findings_inspector

    findings_inspector(findings)


def test_findings_browser_does_not_let_underscores_become_bold():
    f = finding(1, text="Public API is exported from src/click/__init__.py", evidence="src/click/__init__.py")
    at = AppTest.from_function(_escaped_script, args=([f],), default_timeout=15).run()
    assert not at.exception
    shown = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any("src/click/\\_\\_init\\_\\_.py" in v for v in shown)
    assert not any("src/click/__init__.py" in v for v in shown)


def test_report_narrative_headings_sit_below_the_report_title():
    at = AppTest.from_function(_viewer_script, args=("# md", "## Executive summary\nFine [#1] in src/__init__.py.", STATUSES[:1], SAMPLE[:1]), default_timeout=15).run()
    narrative = next(m.value for m in at.markdown if "Executive summary" in m.value)
    assert narrative.startswith("#### Executive summary") and ":gray[`#1`]" in narrative and "src/\\_\\_init\\_\\_.py" in narrative
