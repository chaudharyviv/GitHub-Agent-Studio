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
        reply("Recording.", [finding_call("structure", f"{cls.title} finding", "warning")]),
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
