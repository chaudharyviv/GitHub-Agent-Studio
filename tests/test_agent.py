"""Tests for the single agent loop and toolbox, using a scripted fake OpenAI client and a fake GitHub."""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from agents.single import SingleAgent
from agents.toolbox import Toolbox
from memory import MemoryStore, UserContext
from tools.client import GitHubClient, set_client


def reply(content=None, calls=()):
    tool_calls = [
        SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ] or None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeOpenAI:
    """Returns scripted replies in order and records each request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))  # snapshot: messages list is mutated later
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def github(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/repos/octo/demo":
        return httpx.Response(200, json={
            "name": "demo", "owner": {"login": "octo"}, "description": "A demo", "stargazers_count": 7,
            "language": "Python", "topics": [], "license": None, "default_branch": "main",
            "created_at": "t", "updated_at": "t", "fork": False,
        })
    if path == "/repos/octo/demo/contents/README.md":
        return httpx.Response(200, json={"type": "file", "path": "README.md", "size": 6, "encoding": "base64",
                                         "content": base64.b64encode(b"# Demo").decode(), "sha": "s"})
    return httpx.Response(404, json={"message": "Not Found"})


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


# -- toolbox ---------------------------------------------------------------------

def test_specs_hide_owner_and_repo(store):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    by_name = {s["function"]["name"]: s["function"] for s in box.specs()}
    assert {"get_repository", "get_file_content", "save_finding", "recall_findings", "save_user_context"} <= set(by_name)
    for spec in by_name.values():
        assert "owner" not in spec["parameters"]["properties"] and "repo" not in spec["parameters"]["properties"]
        assert "owner" not in spec["parameters"].get("required", [])
    assert by_name["get_file_content"]["parameters"]["required"] == ["path"]


def test_toolbox_include_subset(store):
    box = Toolbox("octo", "demo", store, "x", include=["get_repository"])
    assert [s["function"]["name"] for s in box.specs()] == ["get_repository"]


def test_repo_is_bound_even_if_model_overrides_it(store):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    outcome = box.call("get_repository", json.dumps({"owner": "evil", "repo": "other"}))
    assert not outcome.is_error and outcome.data["name"] == "demo"


def test_get_repository_saves_profile(store):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    outcome = box.call("get_repository", "{}")
    assert outcome.memory_note == "Saved repository profile"
    assert store.get_repository_profile("octo", "demo").stars == 7


@pytest.mark.parametrize("name,raw,fragment", [
    ("nope", "{}", "unknown tool"),
    ("get_file_content", "{not json", "Bad arguments"),
    ("get_file_content", "{}", "Bad arguments"),  # missing required path
    ("get_file_content", "[]", "JSON object"),
    ("save_finding", json.dumps({"severity": "dire", "category": "c", "finding": "f"}), "Bad arguments"),
])
def test_bad_calls_become_errors_not_exceptions(store, name, raw, fragment):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    outcome = box.call(name, raw)
    assert outcome.is_error and fragment in outcome.data["message"]


def test_github_errors_pass_through_as_tool_errors(store):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    outcome = box.call("get_file_content", json.dumps({"path": "missing.md"}))
    assert outcome.is_error and outcome.data["kind"] == "not_found"


def test_long_results_are_capped(store, monkeypatch):
    monkeypatch.setattr("agents.toolbox.MAX_TOOL_CHARS", 50)
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    assert "[truncated" in box.call("get_repository", "{}").content


def test_save_finding_dedupes_and_recall_filters(store):
    box = Toolbox("octo", "demo", store, store.create_session("octo/demo", "single_agent"))
    args = json.dumps({"severity": "warning", "category": "dependency_risk", "finding": "Old lib", "evidence": "package.json"})
    first = box.call("save_finding", args)
    assert first.memory_note.startswith("Saved finding #")
    second = box.call("save_finding", args)
    assert second.memory_note is None and "Already recorded" in second.data["message"]

    box.call("save_finding", json.dumps({"severity": "info", "category": "architecture", "finding": "Monolith"}))
    recalled = box.call("recall_findings", json.dumps({"category": "dependency_risk"})).data["findings"]
    assert [f["finding"] for f in recalled] == ["Old lib"] and "package.json" in recalled[0]["evidence"]


# -- agent loop ----------------------------------------------------------------------

def test_full_investigation_loop(store):
    llm = FakeOpenAI(
        reply("I'll start with the metadata.", [("get_repository", {})]),
        reply("Now the README.", [("get_file_content", {"path": "README.md"}), ("save_finding", {"severity": "info", "category": "docs", "finding": "Has a README"})]),
        reply("It is a small demo project."),
    )
    result = SingleAgent(store, client=llm, model="gpt-4o-mini").investigate("octo", "demo", "Analyze")

    assert result.answer == "It is a small demo project." and result.steps == 2 and result.error is None
    kinds = [e.kind for e in result.events]
    assert kinds == ["reasoning", "tool_call", "tool_result", "memory",
                     "reasoning", "tool_call", "tool_result", "tool_call", "tool_result", "memory", "final"]
    assert result.events[0].content == "I'll start with the metadata."

    # what the model saw on the 3rd call: tool results chained in the right order
    tool_msgs = [m for m in llm.requests[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_0", "call_1"]
    assert all(r["model"] == "gpt-4o-mini" and r["tools"] for r in llm.requests)

    # memory side effects persisted
    assert [f.finding for f in store.get_findings("octo/demo")] == ["Has a README"]
    history = store.get_conversation_history(result.session_id)
    assert [(m.role, m.content) for m in history] == [("user", "Analyze"), ("assistant", "It is a small demo project.")]
    assert len(history[1].tool_calls) == 3


def test_follow_up_uses_history_and_memory(store):
    agent = SingleAgent(store, client=FakeOpenAI(reply("First answer.")), model="m")
    sid = agent.start_session("octo", "demo")
    store.save_user_context(UserContext(repository_id="octo/demo", key="learning_focus", value="the auth module"))
    agent.investigate("octo", "demo", "Remember I am learning auth", sid)

    llm = FakeOpenAI(reply("Second answer."))
    SingleAgent(store, client=llm, model="m").investigate("octo", "demo", "And now?", sid)
    messages = llm.requests[0]["messages"]
    assert "learning_focus: the auth module" in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == ["Remember I am learning auth", "First answer.", "And now?"]


def test_new_session_sees_findings_from_previous_session(store):
    agent = SingleAgent(store, client=FakeOpenAI(
        reply(None, [("save_finding", {"severity": "warning", "category": "deps", "finding": "Pinned to an old lodash"})]),
        reply("Saved."),
    ), model="m")
    agent.investigate("octo", "demo", "Check deps")

    llm = FakeOpenAI(reply("Yes, from memory."))
    SingleAgent(store, client=llm, model="m").investigate("octo", "demo", "What did you find last time?")
    assert "Pinned to an old lodash" in llm.requests[0]["messages"][0]["content"]


def test_step_limit_forces_an_answer_without_tools(store):
    llm = FakeOpenAI(
        reply("again", [("get_repository", {})]),
        reply("again", [("get_repository", {})]),
        reply("Best effort answer."),
    )
    result = SingleAgent(store, client=llm, model="m", max_steps=2).investigate("octo", "demo", "Go")
    assert result.answer == "Best effort answer." and result.steps == 2
    assert "tools" not in llm.requests[2] and "Tool budget used up" in llm.requests[2]["messages"][-1]["content"]


def test_llm_failure_becomes_error_event(store):
    class AuthenticationError(Exception):
        pass

    result = SingleAgent(store, client=FakeOpenAI(AuthenticationError("bad key")), model="m").investigate("octo", "demo", "Go")
    assert result.error and "OPENAI_API_KEY" in result.error and result.answer == ""


def test_run_streams_events_lazily(store):
    agent = SingleAgent(store, client=FakeOpenAI(reply("Checking.", [("get_repository", {})]), reply("Done.")), model="m")
    stream = agent.run("octo", "demo", "Go", agent.start_session("octo", "demo"))
    assert next(stream).kind == "reasoning"  # nothing beyond the first event has happened yet
    assert next(stream).kind == "tool_call"
    assert store.get_repository_profile("octo", "demo") is None  # tool not executed yet
    assert next(stream).kind == "tool_result"
    assert store.get_repository_profile("octo", "demo") is not None


# -- output token cap --------------------------------------------------------------

def test_every_llm_call_carries_the_token_cap(store):
    llm = FakeOpenAI(reply("Checking.", [("get_repository", {})]), reply("Done."))
    SingleAgent(store, client=llm, model="m", max_output_tokens=777).investigate("octo", "demo", "Go")
    assert [r["max_completion_tokens"] for r in llm.requests] == [777, 777]


def test_default_token_cap(store):
    llm = FakeOpenAI(reply("Done."))
    SingleAgent(store, client=llm, model="m").investigate("octo", "demo", "Go")
    assert llm.requests[0]["max_completion_tokens"] == 2048


def test_truncated_answer_is_flagged(store):
    cut = reply("A long answer that was cu")
    cut.choices[0].finish_reason = "length"
    result = SingleAgent(store, client=FakeOpenAI(cut), model="m").investigate("octo", "demo", "Go")
    assert result.answer.startswith("A long answer that was cu") and "cut off" in result.answer
    assert "cut off" in store.get_conversation_history(result.session_id)[-1].content


def test_tool_calls_on_the_tool_free_final_step_are_not_executed(store):
    llm = FakeOpenAI(
        reply("again", [("get_repository", {})]),
        reply("Answering despite the model asking for a tool.", [("get_repository", {})]),
    )
    result = SingleAgent(store, client=llm, model="m", max_steps=1).investigate("octo", "demo", "Go")
    assert result.answer == "Answering despite the model asking for a tool." and result.error is None
