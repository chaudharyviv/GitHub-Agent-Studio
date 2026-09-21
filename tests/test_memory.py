"""Tests for the SQLite memory layer."""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memory import ConversationMessage, Finding, MemoryStore, RepositoryProfile, UserContext

REPO = "Octo/Demo"  # mixed case on purpose: ids are case-insensitive


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def finding(session_id, **overrides):
    data = dict(session_id=session_id, agent="security", severity="warning", category="dependency_risk",
                finding="Outdated lib", evidence_json='{"file": "package.json"}', confidence=0.8)
    return Finding(**{**data, **overrides})


def profile(**overrides):
    data = dict(owner="Octo", name="Demo", stars=5, default_branch="main", primary_language="Python")
    return RepositoryProfile(**{**data, **overrides})


def test_findings_roundtrip_and_repo_id_normalisation(store):
    sid = store.create_session(REPO, "multi_agent")
    fid = store.save_finding(finding(sid))
    [got] = store.get_findings("octo/demo")
    assert got.id == fid and got.repo_id == "octo/demo" and got.session_id == sid
    assert got.confidence == 0.8 and json.loads(got.evidence_json) == {"file": "package.json"}
    assert got.created_at.tzinfo is not None


def test_findings_filters_and_order(store):
    s1 = store.create_session(REPO, "single_agent")
    s2 = store.create_session(REPO, "multi_agent")
    store.save_finding(finding(s1, agent="single_agent", severity="info", category="architecture", finding="one"))
    store.save_finding(finding(s2, agent="security", severity="critical", category="secrets", finding="two"))
    store.save_finding(finding(s2, agent="quality", severity="warning", category="tests", finding="three"))

    assert [f.finding for f in store.get_findings(REPO)] == ["three", "two", "one"]  # newest first
    assert [f.finding for f in store.get_findings(REPO, agent="security")] == ["two"]
    assert [f.finding for f in store.get_findings(REPO, severity="info")] == ["one"]
    assert [f.finding for f in store.get_findings(REPO, category="tests")] == ["three"]
    assert [f.finding for f in store.get_findings(REPO, session_id=s1)] == ["one"]
    assert len(store.get_findings(REPO, limit=2)) == 2


def test_findings_are_scoped_per_repository(store):
    a = store.create_session("a/one", "single_agent")
    b = store.create_session("b/two", "single_agent")
    store.save_finding(finding(a, finding="for a"))
    store.save_finding(finding(b, finding="for b"))
    assert [f.finding for f in store.get_findings("a/one")] == ["for a"]


def test_finding_requires_known_session_and_valid_fields(store):
    with pytest.raises(ValueError):
        store.save_finding(finding("nope"))
    with pytest.raises(ValidationError):
        finding("x", severity="catastrophic")
    with pytest.raises(ValidationError):
        finding("x", confidence=1.5)
    with pytest.raises(ValidationError):
        finding("x", evidence_json="not json")


def test_invalid_repo_id_and_mode(store):
    for bad in ("nodash", "a/b/c", "/x", "x/"):
        with pytest.raises(ValueError):
            store.get_findings(bad)
    with pytest.raises(ValueError):
        store.create_session(REPO, "swarm")


def test_profile_upsert_and_lookup(store):
    assert store.get_repository_profile("Octo", "Demo") is None
    store.save_repository_profile(profile(stars=5))
    store.save_repository_profile(profile(stars=99, topics=["ai"]))
    got = store.get_repository_profile("octo", "demo")
    assert got.stars == 99 and got.topics == ["ai"] and got.owner == "Octo"
    assert store.list_repositories() == ["octo/demo"]


def test_user_context_upsert_and_delete(store):
    store.save_user_context(UserContext(repository_id=REPO, key="learning_focus", value="auth module"))
    store.save_user_context(UserContext(repository_id=REPO, key="learning_focus", value="billing"))
    store.save_user_context(UserContext(repository_id=REPO, key="language", value="en"))
    got = {c.key: c.value for c in store.get_user_context(REPO)}
    assert got == {"learning_focus": "billing", "language": "en"}
    assert store.delete_user_context(REPO, "language") is True
    assert store.delete_user_context(REPO, "language") is False


def test_conversation_history_order_limit_and_tool_calls(store):
    sid = store.create_session(REPO, "single_agent")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, role in enumerate(["user", "assistant", "user"]):
        store.save_conversation_message(ConversationMessage(
            session_id=sid, role=role, content=f"m{i}", timestamp=base + timedelta(seconds=i),
            tool_calls=['{"name": "get_repository"}'] if role == "assistant" else None,
        ))
    history = store.get_conversation_history(sid)
    assert [m.content for m in history] == ["m0", "m1", "m2"]
    assert history[1].tool_calls == ['{"name": "get_repository"}'] and history[0].tool_calls is None
    assert [m.content for m in store.get_conversation_history(sid, limit=2)] == ["m1", "m2"]
    with pytest.raises(ValueError):
        store.save_conversation_message(ConversationMessage(session_id="nope", role="user", content="x"))


def test_sessions(store):
    s1 = store.create_session(REPO, "single_agent", metadata='{"k": 1}')
    s2 = store.create_session(REPO, "multi_agent")
    assert [s.session_id for s in store.list_sessions(REPO)] == [s2, s1]
    assert [s.session_id for s in store.list_sessions(REPO, mode="single_agent")] == [s1]
    assert store.get_session(s1).metadata == '{"k": 1}' and store.get_session(s1).completed_at is None
    store.complete_session(s1)
    assert store.get_session(s1).completed_at is not None
    assert store.get_session("nope") is None
    with pytest.raises(ValueError):
        store.complete_session("nope")


def test_resume_context(store):
    empty = store.get_resume_context(REPO)
    assert empty.last_session is None and empty.findings == [] and empty.profile is None

    store.save_repository_profile(profile())
    s1 = store.create_session(REPO, "single_agent")
    store.save_finding(finding(s1, finding="old"))
    store.save_conversation_message(ConversationMessage(session_id=s1, role="user", content="hi"))
    s2 = store.create_session(REPO, "multi_agent")
    store.save_finding(finding(s2, finding="new"))
    store.save_user_context(UserContext(repository_id=REPO, key="learning_focus", value="auth"))

    ctx = store.get_resume_context(REPO)
    assert ctx.last_session.session_id == s2
    assert [f.finding for f in ctx.findings] == ["new", "old"]
    assert [m.content for m in ctx.recent_messages] == ["hi"]
    assert ctx.user_context[0].value == "auth" and ctx.profile.stars == 5


def test_clear_repository_memory_removes_only_that_repo(store):
    for repo in ("a/one", "b/two"):
        sid = store.create_session(repo, "single_agent")
        store.save_finding(finding(sid))
        store.save_conversation_message(ConversationMessage(session_id=sid, role="user", content="x"))
        store.save_user_context(UserContext(repository_id=repo, key="k", value="v"))
        owner, name = repo.split("/")
        store.save_repository_profile(profile(owner=owner, name=name))

    store.clear_repository_memory("A/One")
    assert store.get_findings("a/one") == [] and store.list_sessions("a/one") == []
    assert store.get_user_context("a/one") == [] and store.get_repository_profile("a", "one") is None
    assert len(store.get_findings("b/two")) == 1 and store.list_repositories() == ["b/two"]
    store.clear_repository_memory("a/one")  # clearing nothing is fine


def test_export(store):
    sid = store.create_session(REPO, "single_agent")
    store.save_finding(finding(sid))
    store.save_conversation_message(ConversationMessage(session_id=sid, role="user", content="hi"))
    store.save_user_context(UserContext(repository_id=REPO, key="k", value="v"))
    data = store.export_repository_memory(REPO)
    json.dumps(data)  # must be JSON-serializable
    assert data["repo_id"] == "octo/demo"
    assert len(data["findings"]) == len(data["sessions"]) == len(data["messages"]) == len(data["user_context"]) == 1


def test_persists_across_restart(tmp_path):
    path = str(tmp_path / "mem.db")
    first = MemoryStore(path)
    sid = first.create_session(REPO, "single_agent")
    first.save_finding(finding(sid, finding="survives"))
    first.save_user_context(UserContext(repository_id=REPO, key="k", value="v"))
    first.close()

    second = MemoryStore(path)
    assert [f.finding for f in second.get_findings(REPO)] == ["survives"]
    assert second.get_user_context(REPO)[0].value == "v"
    assert second.get_resume_context(REPO).last_session.session_id == sid
    second.close()


def test_rejects_database_from_newer_version(tmp_path):
    path = str(tmp_path / "future.db")
    MemoryStore(path).close()
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        MemoryStore(path)


def test_concurrent_writes_from_threads(store):
    sid = store.create_session(REPO, "multi_agent")
    errors = []

    def work(n):
        try:
            for i in range(20):
                store.save_finding(finding(sid, finding=f"{n}-{i}"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(store.get_findings(REPO)) == 80
