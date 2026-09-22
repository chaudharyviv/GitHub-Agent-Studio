"""Tests for the four War Room specialists (scripted fake LLM, fake GitHub)."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from agents.multi import (
    SPECIALISTS,
    ArchitectureSpecialist,
    HealthSpecialist,
    QualitySpecialist,
    SecuritySpecialist,
)
from memory import MemoryStore
from tests.test_agent import FakeOpenAI, github, reply
from tools.client import GitHubClient, set_client


@pytest.fixture(autouse=True)
def fake_github():
    set_client(GitHubClient(transport=httpx.MockTransport(github), sleep=lambda _: None))
    yield
    set_client(None)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def finding_call(category, text, severity="info", evidence="README.md"):
    return ("save_finding", {"severity": severity, "category": category, "finding": text, "evidence": evidence})


def tool_names(request):
    return {t["function"]["name"] for t in request["tools"]}


def test_each_specialist_has_its_own_identity_and_tools():
    ids = [cls.agent_id for cls in SPECIALISTS]
    assert ids == ["architecture_specialist", "health_specialist", "quality_specialist", "security_specialist"]
    assert "get_issues" in HealthSpecialist.tools and "get_issues" not in ArchitectureSpecialist.tools
    assert "search_code" in SecuritySpecialist.tools and "search_code" not in QualitySpecialist.tools
    assert "get_repository_tree" not in HealthSpecialist.tools


@pytest.mark.parametrize("cls,title,own_category,foreign_category", [
    (ArchitectureSpecialist, "Architecture Specialist", "entry_points", "bus_factor"),
    (SecuritySpecialist, "Security Specialist", "dependency_risk", "bus_factor"),
    (QualitySpecialist, "Code Quality Specialist", "error_handling", "bus_factor"),
    (HealthSpecialist, "Project Health Specialist", "bus_factor", "entry_points"),
])
def test_prompts_are_focused_and_dated(cls, title, own_category, foreign_category):
    prompt = cls.prompt(8)
    assert title in prompt and "save_finding" in prompt and "at most 8 tool-calling rounds" in prompt
    assert f"{datetime.now(timezone.utc):%Y-%m-%d}" in prompt
    assert own_category in prompt and foreign_category not in prompt


@pytest.mark.parametrize("cls", SPECIALISTS)
def test_specialist_only_sees_its_tools_and_saves_attributed_findings(store, cls):
    llm = FakeOpenAI(
        reply("Recording.", [finding_call(cls.categories[0], f"{cls.title} finding", "warning")]),
        reply("Covered the basics."),
    )
    result = cls(store, client=llm, model="m").investigate("octo", "demo")

    assert result.error is None and result.answer == "Covered the basics."
    assert tool_names(llm.requests[0]) == set(cls.tools) | {"save_finding", "recall_findings"}
    assert "save_user_context" not in tool_names(llm.requests[0])
    [saved] = result.findings
    assert saved.agent == cls.agent_id and saved.finding == f"{cls.title} finding" and saved.severity == "warning"
    assert json.loads(saved.evidence_json) == {"evidence": "README.md"}
    assert store.get_session(result.session_id).mode == "multi_agent"


def test_specialist_without_findings_is_nudged_once(store):
    llm = FakeOpenAI(
        reply("Looked at everything. Nothing to report."),  # finishes with nothing saved
        reply("Saving now.", [finding_call("activity", "Nothing notable found", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    assert [f.finding for f in result.findings] == ["Nothing notable found"] and result.answer == "Done."
    nudge = llm.requests[1]["messages"][-1]
    assert nudge["role"] == "user" and "without recording any findings" in nudge["content"]
    assert any("asking for them" in e.content for e in result.events if e.kind == "reasoning")


def test_nudge_only_happens_once(store):
    llm = FakeOpenAI(reply("Nothing."), reply("Still nothing."))
    result = HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    assert result.findings == [] and result.answer == "Still nothing." and len(llm.requests) == 2


def test_specialist_sees_teammates_findings_in_shared_memory(store):
    session = store.create_session("octo/demo", "multi_agent")
    ArchitectureSpecialist(store, client=FakeOpenAI(
        reply(None, [finding_call("tech_stack", "Uses Django 4 and Celery", "info", "requirements.txt")]),
        reply("Done."),
    ), model="m").investigate("octo", "demo", session_id=session)

    llm = FakeOpenAI(reply(None, [finding_call("dependency_risk", "Django is unpinned", "warning")]), reply("Done."))
    result = SecuritySpecialist(store, client=llm, model="m").investigate("octo", "demo", session_id=session)

    system = llm.requests[0]["messages"][0]["content"]
    assert "[architecture_specialist] info/tech_stack: Uses Django 4 and Celery" in system
    assert [f.agent for f in result.findings] == ["security_specialist"]  # only its own
    assert {f.agent for f in store.get_findings("octo/demo", session_id=session)} == {"architecture_specialist", "security_specialist"}


def test_own_and_other_sessions_findings_are_not_shared(store):
    old = store.create_session("octo/demo", "multi_agent")
    ArchitectureSpecialist(store, client=FakeOpenAI(reply(None, [finding_call("structure", "Old run")]), reply("Done.")), model="m").investigate(
        "octo", "demo", session_id=old)
    llm = FakeOpenAI(reply(None, [finding_call("activity", "New run")]), reply("Done."))
    HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")  # brand-new session
    assert "Old run" not in llm.requests[0]["messages"][0]["content"]
    assert "You are the first specialist" in llm.requests[0]["messages"][0]["content"]


def test_user_focus_is_passed_through(store):
    llm = FakeOpenAI(reply(None, [finding_call("secrets", "x")]), reply("Done."))
    SecuritySpecialist(store, client=llm, model="m").investigate("octo", "demo", query="focus on GitHub Actions")
    assert "focus on GitHub Actions" in llm.requests[0]["messages"][1]["content"]


def test_llm_failure_is_reported_not_raised(store):
    class RateLimitError(Exception):
        pass

    result = QualitySpecialist(store, client=FakeOpenAI(RateLimitError("slow down")), model="m").investigate("octo", "demo")
    assert result.error and "rate limit" in result.error and result.findings == []


def test_uses_specialist_token_cap_and_step_limit(store):
    llm = FakeOpenAI(
        reply("a", [("get_repository", {})]),
        reply("b", [finding_call("activity", "Partial view", "warning")]),
        reply("Ran out of steps; did not check releases."),  # third call is the tool-free forced answer
    )
    result = HealthSpecialist(store, client=llm, model="m", max_steps=2).investigate("octo", "demo")
    assert all(r["max_completion_tokens"] == 1536 for r in llm.requests)
    assert "tools" not in llm.requests[2] and "Tool budget used up" in llm.requests[2]["messages"][-1]["content"]
    assert result.answer.startswith("Ran out of steps") and result.steps == 2 and len(result.findings) == 1


def test_each_specialist_has_its_own_category_list():
    assert ArchitectureSpecialist.categories == ("structure", "entry_points", "modules", "patterns", "tech_stack")
    assert "bus_factor" in HealthSpecialist.categories and "bus_factor" not in SecuritySpecialist.categories
    for cls in SPECIALISTS:  # the prompt tells the model exactly the list the toolbox enforces
        assert ", ".join(cls.categories) in cls.prompt(8)


def test_categories_outside_the_specialists_lane_are_rejected_and_the_model_can_retry(store):
    llm = FakeOpenAI(
        reply(None, [finding_call("documentation", "README is nice", "info")]),  # not a security category
        reply(None, [finding_call("security_policy", "No SECURITY.md in root or .github/", "warning")]),
        reply("Done."),
    )
    result = SecuritySpecialist(store, client=llm, model="m").investigate("octo", "demo")
    rejection = json.loads(next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")["content"])
    assert rejection["kind"] == "invalid_input" and "security_policy" in rejection["message"]
    assert [f.finding for f in result.findings] == ["No SECURITY.md in root or .github/"]


def test_single_agent_categories_stay_unrestricted(store):
    from agents.toolbox import Toolbox

    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    outcome = box.call("save_finding", json.dumps({"severity": "info", "category": "anything_goes", "finding": "x"}))
    assert not outcome.is_error


def test_identical_github_calls_are_not_repeated(store):
    llm = FakeOpenAI(
        reply("Fetching.", [("get_repository", {}), ("get_repository", {})]),  # same call twice in one round
        reply("Again.", [("get_repository", {})]),
        reply(None, [finding_call("activity", "Checked metadata", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    results = [e for e in result.events if e.kind == "tool_result"]
    assert [e.is_error for e in results] == [False, False, False, False]
    assert "note" not in results[0].data and all("already fetched" in e.data["note"] for e in results[1:3])
    assert [e.content.startswith("ok") for e in results[:3]] == [True, True, True]


def test_failed_github_calls_may_be_retried(store):
    from agents.toolbox import Toolbox

    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    args = json.dumps({"path": "missing.md"})
    assert box.call("get_file_content", args).data["kind"] == "not_found"
    assert box.call("get_file_content", args).data["kind"] == "not_found"  # asked again, not short-circuited


def test_save_reminder_is_injected_when_two_rounds_remain(store):
    llm = FakeOpenAI(
        reply("r1", [("get_repository", {})]),
        reply("r2", [("get_commits", {"limit": 5})]),
        reply("r3", [("get_releases", {"limit": 5})]),
        reply(None, [finding_call("activity", "Reminded in time", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m", max_steps=4).investigate("octo", "demo")
    seen = lambda i: any("round(s) left" in m.get("content", "") for m in llm.requests[i]["messages"] if m["role"] == "system")
    assert [seen(i) for i in range(4)] == [False, False, True, True]  # appears once 2 rounds remain
    assert sum("round(s) left" in m.get("content", "") for m in llm.requests[3]["messages"]) == 1
    assert len(result.findings) == 1


def test_pull_requests_is_a_health_category_and_rejections_say_how_to_recover(store):
    assert "pull_requests" in HealthSpecialist.categories
    llm = FakeOpenAI(
        reply(None, [finding_call("made_up_category", "x", "info")]),
        reply(None, [finding_call("pull_requests", "10 open PRs; oldest from 2026-08-19", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    rejection = json.loads(next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")["content"])
    assert "Call save_finding again with the same finding" in rejection["message"]
    assert [f.category for f in result.findings] == ["pull_requests"]


# -- guards that do not depend on the model obeying the prompt ------------------------------

def test_final_tool_round_only_offers_the_memory_tools(store):
    llm = FakeOpenAI(
        reply("r1", [("get_repository", {})]),
        reply("r2", [("get_commits", {"limit": 5})]),  # max_steps=3 -> the third round is save-only
        reply(None, [finding_call("activity", "Saved in the last round", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m", max_steps=3).investigate("octo", "demo")
    assert "get_repository" in tool_names(llm.requests[0]) and "get_repository" in tool_names(llm.requests[1])
    assert tool_names(llm.requests[2]) == {"save_finding", "recall_findings"}
    assert len(result.findings) == 1


def test_nudge_attempt_can_only_save_and_uses_few_rounds(store):
    llm = FakeOpenAI(
        reply("Nothing to report."),  # done, nothing saved
        reply(None, [finding_call("activity", "Saved after the nudge", "info")]),
        reply("Done."),
    )
    result = HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    assert "get_issues" in tool_names(llm.requests[0])
    assert tool_names(llm.requests[1]) == {"save_finding", "recall_findings"}
    assert len(result.findings) == 1


def test_findings_written_as_prose_trigger_a_correction(store):
    essay = "Here are my findings: 1. No SECURITY.md. 2. Missing lockfile. " * 12  # far longer than 400 chars
    llm = FakeOpenAI(
        reply(essay, [("get_repository", {})]),
        reply(None, [finding_call("activity", "No SECURITY.md (from the essay)", "warning")]),
        reply("Done."),
    )
    HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    second_request = llm.requests[1]["messages"]
    assert "Prose is discarded" in second_request[-1]["content"] and second_request[-1]["role"] == "system"
    assert not any("Prose is discarded" in (m.get("content") or "") for m in llm.requests[0]["messages"])


def test_short_reasoning_gets_no_correction(store):
    llm = FakeOpenAI(reply("Checking metadata.", [("get_repository", {})]), reply(None, [finding_call("activity", "x")]), reply("Done."))
    HealthSpecialist(store, client=llm, model="m").investigate("octo", "demo")
    assert not any("Prose is discarded" in (m.get("content") or "") for r in llm.requests for m in r["messages"])
