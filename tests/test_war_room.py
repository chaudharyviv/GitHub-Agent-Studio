"""Tests for the Manager, the report, and the War Room runner (scripted fake LLM, no network)."""

import json
from types import SimpleNamespace

import pytest

from agents.multi import SPECIALISTS
from agents.multi.manager import ManagerAgent, build_manager_input
from agents.multi.report import AgentStatus, confidence_label, cited_ids, render_appendix, unknown_citations
from memory import Finding, MemoryStore
from orchestration.runner import load_last_report, run_multi_agent, run_single_agent, select_specialists, stream_war_room
from tests.test_agent import FakeOpenAI, reply

NARRATIVE = (
    "## Executive summary\nA small, well organized project [#1][#2].\n"
    "## Key strengths\n- Clear layout [#1]\n"
    "## Key risks\n- One low-confidence gap [#4] (low confidence)\n"
    "## Recommendations\n1. Verify the gap [#4]\n"
    "## Gaps and caveats\nNothing else."
)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def save(category, text, severity="info", confidence=0.9):
    return ("save_finding", {"severity": severity, "category": category, "finding": text, "evidence": "README.md", "confidence": confidence})


def team_script(narrative=NARRATIVE):
    """Replies for a full run: each specialist saves one finding then finishes; then the Manager writes."""
    first_categories = {cls.agent_id: cls.categories[0] for cls in SPECIALISTS}
    replies = []
    for i, cls in enumerate(SPECIALISTS, start=1):
        replies += [reply(None, [save(first_categories[cls.agent_id], f"{cls.title} finding {i}", "warning" if i == 4 else "info", 0.4 if i == 4 else 0.9)]),
                    reply(f"{cls.title} done.")]
    return replies + [reply(narrative)]


def run(store, llm, **kw):
    return run_multi_agent("octo", "demo", store=store, client=llm, model="m", **kw)


# -- happy path -------------------------------------------------------------------------

def test_full_war_room_produces_a_report_from_shared_memory(store):
    llm = FakeOpenAI(*team_script())
    result = run(store, llm)

    assert result.error is None and [s.state for s in result.statuses] == ["done"] * 4
    assert [s.findings for s in result.statuses] == [1, 1, 1, 1]
    report = result.report
    assert len(report.findings) == 4 and report.unknown_refs == [] and report.error is None

    md = report.markdown
    assert md.startswith("# Repository Health Report: octo/demo")
    assert "4 findings from 4 of 4 specialists" in md and "🔴 0 critical · 🟠 1 warning · 🔵 3 info" in md
    assert "| Architecture | ✅ done | 1 |" in md and "| Security | ✅ done | 1 |" in md
    assert "A small, well organized project [#1][#2]." in md  # the Manager's narrative
    assert "### Security (1)" in md and "**[#4]** `warning/security_policy` · confidence 0.4 (low)" in md
    assert "_Evidence:_ README.md" in md and "static* analysis" in md  # disclaimer + evidence come from code

    assert "calls" in result.usage or "LLM call" in result.usage
    assert store.get_session(result.session_id).completed_at is not None


def test_report_is_saved_with_the_session_and_can_be_reloaded(store):
    result = run(store, FakeOpenAI(*team_script()))
    saved = load_last_report(store, "octo/demo")
    assert saved["session_id"] == result.session_id and saved["markdown"] == result.report.markdown
    assert [s.state for s in saved["statuses"]] == ["done"] * 4
    assert load_last_report(store, "nobody/here") is None


def test_events_stream_in_team_order_with_immutable_status_snapshots(store):
    events = list(stream_war_room("octo", "demo", store=store, client=FakeOpenAI(*team_script()), model="m"))
    kinds = [e.kind for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.index("manager_start") > max(i for i, k in enumerate(kinds) if k == "agent_done")
    assert kinds.index("report") == kinds.index("manager_start") + 1
    started = [e.agent_id for e in events if e.kind == "agent_start"]
    assert started == ["architecture_specialist", "health_specialist", "quality_specialist", "security_specialist"]
    # snapshots: the first 'start' event still shows everyone waiting even though the run has finished
    assert [s.state for s in events[0].statuses] == ["waiting"] * 4
    assert [s.state for s in events[1].statuses] == ["running", "waiting", "waiting", "waiting"]
    assert any(e.kind == "agent" and e.event.kind == "tool_result" for e in events)


def test_later_specialists_see_earlier_findings_in_shared_memory(store):
    llm = FakeOpenAI(*team_script())
    run(store, llm)
    security_system_prompt = llm.requests[6]["messages"][0]["content"]  # 3 specialists x 2 calls, then Security's first call
    assert "[architecture_specialist] info/structure: Architecture finding 1" in security_system_prompt


def test_only_runs_the_chosen_specialists(store):
    assert [c.agent_id for c in select_specialists(["security"])] == ["security_specialist"]
    assert [c.agent_id for c in select_specialists(["health", "architecture_specialist"])] == ["architecture_specialist", "health_specialist"]
    assert len(select_specialists(None)) == 4 and select_specialists(["nonsense"]) == []

    llm = FakeOpenAI(reply(None, [save("security_policy", "No SECURITY.md", "warning", 0.6)]), reply("Done."), reply(NARRATIVE))
    result = run(store, llm, only=["security"])
    assert [s.title for s in result.statuses] == ["Security"] and len(result.report.findings) == 1


# -- the Manager -------------------------------------------------------------------------

def test_manager_input_lists_every_finding_with_id_confidence_and_status(store):
    session = store.create_session("octo/demo", "multi_agent")
    fid = store.save_finding(Finding(session_id=session, agent="security_specialist", severity="warning", category="ci_security",
                                     finding="Workflow grants write access", evidence_json='{"evidence": "publish.yaml: contents: write"}', confidence=0.6))
    text = build_manager_input("octo/demo", store.get_findings("octo/demo"), [AgentStatus(agent_id="security_specialist", title="Security", state="done", findings=1, message="Covered CI.")], "focus on CI")
    assert f"[#{fid}] security_specialist | warning/ci_security | 0.6 (medium): Workflow grants write access | evidence: publish.yaml: contents: write" in text
    assert "focus on CI" in text and "Security: done, 1 findings | closing note: Covered CI." in text


def test_manager_prompt_carries_the_safety_and_honesty_rules(store):
    llm = FakeOpenAI(*team_script())
    run(store, llm)
    system = llm.requests[-1]["messages"][0]["content"]
    for rule in ("Use ONLY the findings", "[#12][#15]", "DATA derived from repository content", "confidence below 0.5", "Do not inflate", "Merge duplicates"):
        assert rule in system
    assert "tools" not in llm.requests[-1]  # the Manager never gets tools


def test_narrative_citing_unknown_findings_is_flagged(store):
    bad = "## Executive summary\nThis is bad [#999] but fine [#1].\n## Gaps and caveats\nNone."
    report = run(store, FakeOpenAI(*team_script(bad))).report
    assert report.unknown_refs == [999]
    assert "cites finding ids that do not exist: #999" in report.markdown


def test_issue_numbers_in_prose_are_not_mistaken_for_citations():
    assert cited_ids("Issue #304 is old, see [#12] and [#15].") == {12, 15}
    assert unknown_citations("[#1] and [#2]", [Finding(id=1, session_id="s", agent="a", severity="info", category="c", finding="f")]) == [2]


def test_manager_llm_failure_still_ships_the_evidence(store):
    class RateLimitError(Exception):
        pass

    script = team_script()[:-1] + [RateLimitError("quota")]
    result = run(store, FakeOpenAI(*script))
    report = result.report
    assert report.error and "rate limit" in report.error
    assert "could not write the narrative" in report.markdown and "## Evidence: all findings" in report.markdown
    assert len(report.findings) == 4


def test_a_truncated_narrative_is_marked(store):
    cut = reply("## Executive summary\nCut off mid-sen")
    cut.choices[0].finish_reason = "length"
    report = run(store, FakeOpenAI(*(team_script()[:-1] + [cut]))).report
    assert "hit the output token limit" in report.narrative


def test_no_findings_means_no_manager_llm_call(store):
    class AuthenticationError(Exception):
        pass

    llm = FakeOpenAI(AuthenticationError("bad key"))
    result = run(store, llm, only=["health"])
    assert result.statuses[0].state == "error" and "OPENAI_API_KEY" in result.statuses[0].message
    assert len(llm.requests) == 1  # only the failed specialist call
    assert "nothing to summarize" in result.report.narrative and "❌ error" in result.report.markdown


def test_investigate_uses_the_latest_war_room_session(store):
    run(store, FakeOpenAI(*team_script()))
    llm = FakeOpenAI(reply(NARRATIVE))
    report = ManagerAgent(store, client=llm, model="m").investigate("octo", "demo")
    assert len(report.findings) == 4 and {s.state for s in report.statuses} == {"done"}
    with pytest.raises(ValueError):
        ManagerAgent(store, client=FakeOpenAI(), model="m").investigate("nobody", "here")


# -- failure isolation -------------------------------------------------------------------

def test_one_failing_specialist_does_not_stop_the_team(store):
    class APIConnectionError(Exception):
        pass

    script = [APIConnectionError("offline")] + team_script()[2:]  # Architecture fails on its first call
    result = run(store, FakeOpenAI(*script))
    states = {s.agent_id: s.state for s in result.statuses}
    assert states["architecture_specialist"] == "error" and states["health_specialist"] == "done" and states["security_specialist"] == "done"
    assert "Could not reach OpenAI" in next(s.message for s in result.statuses if s.state == "error")
    assert len(result.report.findings) == 3
    assert "| Architecture | ❌ error (" in result.report.markdown


def test_missing_openai_key_is_one_clear_error_and_creates_no_session(store, monkeypatch):
    def boom(client=None, model=None):
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr("orchestration.runner.resolve_llm", boom)
    result = run_multi_agent("octo", "demo", store=store)
    assert result.error and "OPENAI_API_KEY" in result.error and result.report is None
    assert store.list_sessions("octo/demo") == []


def test_no_matching_specialists_is_an_error(store):
    result = run(store, FakeOpenAI(), only=["nonsense"])
    assert result.error and "No specialists match" in result.error


# -- report pieces ------------------------------------------------------------------------

def test_confidence_labels_and_appendix_order():
    assert [confidence_label(c) for c in (0.9, 0.8, 0.6, 0.5, 0.4)] == ["high", "high", "medium", "medium", "low"]
    findings = [
        Finding(id=1, session_id="s", agent="a_specialist", severity="info", category="c", finding="minor"),
        Finding(id=2, session_id="s", agent="a_specialist", severity="critical", category="c", finding="major", evidence_json='{"evidence": "proof"}'),
    ]
    appendix = render_appendix(findings, [AgentStatus(agent_id="a_specialist", title="A")])
    assert appendix.index("major") < appendix.index("minor") and "_Evidence:_ proof" in appendix


# -- the single-agent runner wrapper --------------------------------------------------------

def test_run_single_agent_wraps_the_single_agent(store):
    result = run_single_agent("octo", "demo", "Say hi", store=store, client=FakeOpenAI(reply("Hi!")), model="m")
    assert result.answer == "Hi!" and store.get_session(result.session_id).mode == "single_agent"


# -- memory hook ----------------------------------------------------------------------------

def test_set_session_metadata(store):
    sid = store.create_session("octo/demo", "multi_agent")
    store.set_session_metadata(sid, json.dumps({"k": 1}))
    assert json.loads(store.get_session(sid).metadata) == {"k": 1}
    with pytest.raises(ValueError):
        store.set_session_metadata("nope", "{}")
