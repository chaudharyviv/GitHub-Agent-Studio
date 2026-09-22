"""
Agent orchestration and execution runners.

This module provides the entry points:
- run_single_agent: Execute single transparent agent mode
- stream_war_room: Coordinate the specialists + Manager, yielding every step as it happens
- run_multi_agent: The same, collected into one result

The War Room runs the specialists one after another (easy to follow and debug, and later
specialists can read what earlier ones saved), then the Manager writes the report. Specialists
share nothing except the findings in memory. One failing specialist does not stop the others.
"""

import json
from typing import Any, Iterator, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from agents.base import AgentEvent, InvestigationResult
from agents.limits import LITE, NORMAL
from agents.loop import resolve_llm
from agents.multi import SPECIALISTS
from agents.multi.manager import ManagerAgent
from agents.multi.report import AgentStatus, HealthReport
from agents.single import SingleAgent
from agents.usage import UsageMeter
from memory import MemoryStore


class WarRoomEvent(BaseModel):
    """One step of a War Room run, for the UI."""
    kind: Literal["start", "agent_start", "agent", "agent_done", "manager_start", "report", "error", "done"]
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    title: Optional[str] = None
    event: Optional[AgentEvent] = None  # set for kind="agent": what the specialist just did
    statuses: list[AgentStatus] = Field(default_factory=list)  # snapshot of the whole team, on every event that changes it
    report: Optional[HealthReport] = None
    message: str = ""
    usage: str = ""


class WarRoomResult(BaseModel):
    session_id: str
    report: Optional[HealthReport] = None
    statuses: list[AgentStatus] = Field(default_factory=list)
    events: list[WarRoomEvent] = Field(default_factory=list)
    usage: str = ""
    error: Optional[str] = None


def run_single_agent(
    owner: str,
    repo: str,
    user_query: Optional[str] = None,
    *,
    store: Optional[MemoryStore] = None,
    session_id: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
) -> InvestigationResult:
    """
    Execute a single-agent investigation into a repository and return the full result.

    The agent loads earlier findings from memory, calls tools, saves new findings, and answers.
    (For live streaming, use ``SingleAgent.run`` directly.)
    """
    agent = SingleAgent(store or MemoryStore(), client=client, model=model)
    return agent.investigate(owner, repo, user_query, session_id)


def select_specialists(only: Optional[Sequence[str]] = None) -> list:
    """The specialist classes to run, in team order. ``only`` matches agent ids or short names ('health')."""
    if not only:
        return list(SPECIALISTS)
    wanted = [o.lower() for o in only]
    return [cls for cls in SPECIALISTS if any(w == cls.agent_id or w in cls.agent_id for w in wanted)]


def stream_war_room(
    owner: str,
    repo: str,
    user_query: Optional[str] = None,
    *,
    store: Optional[MemoryStore] = None,
    client: Any = None,
    model: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    max_output_tokens: Optional[int] = None,
    lite_mode: Optional[bool] = None,
) -> Iterator[WarRoomEvent]:
    """
    Run the specialists in order, then the Manager, yielding an event for every step.

    The last event is 'done', or 'error' if the run could not start (for example no OpenAI key).
    """
    store = store or MemoryStore()
    classes = select_specialists(only)
    if not classes:
        yield WarRoomEvent(kind="error", message=f"No specialists match {list(only or [])}.")
        return
    try:
        client, model = resolve_llm(client, model)  # one client shared by the whole team
    except Exception as exc:
        yield WarRoomEvent(kind="error", message=f"Could not start the OpenAI client: {exc}")
        return

    repo_id = MemoryStore.make_repo_id(owner, repo)
    session_id = store.create_session(repo_id, "multi_agent")
    statuses = [AgentStatus(agent_id=c.agent_id, title=c.title) for c in classes]
    snapshot = lambda: [s.model_copy() for s in statuses]  # noqa: E731  (events must not change after they are yielded)
    yield WarRoomEvent(kind="start", session_id=session_id, statuses=snapshot())

    manager_kwargs = {"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}
    specialist_kwargs = dict(manager_kwargs)
    if lite_mode is not None:  # explicit override; omitted, each specialist falls back to LITE_MODE from the environment
        specialist_kwargs["limits"] = LITE if lite_mode else NORMAL
    usage = UsageMeter()
    for cls, status in zip(classes, statuses):
        status.state = "running"
        yield WarRoomEvent(kind="agent_start", session_id=session_id, agent_id=cls.agent_id, title=cls.title, statuses=snapshot())

        agent = cls(store, client=client, model=model, **specialist_kwargs)
        last: Optional[AgentEvent] = None
        for event in agent.run(owner, repo, session_id, user_query):
            last = event
            yield WarRoomEvent(kind="agent", session_id=session_id, agent_id=cls.agent_id, title=cls.title, event=event)

        status.findings = len(store.get_findings(repo_id, agent=cls.agent_id, session_id=session_id))
        status.state = "done" if last is not None and last.kind == "final" else "error"
        status.message = last.content if last is not None else "produced no output"
        for field in ("calls", "prompt_tokens", "cached_tokens", "completion_tokens"):
            setattr(usage, field, getattr(usage, field) + getattr(agent.usage, field))
        yield WarRoomEvent(kind="agent_done", session_id=session_id, agent_id=cls.agent_id, title=cls.title,
                           statuses=snapshot(), usage=agent.usage.summary())

    yield WarRoomEvent(kind="manager_start", session_id=session_id, title="Manager", statuses=snapshot())
    manager = ManagerAgent(store, client=client, model=model, **manager_kwargs)
    report = manager.synthesize(owner, repo, session_id, statuses, user_query)
    for field in ("calls", "prompt_tokens", "cached_tokens", "completion_tokens"):
        setattr(usage, field, getattr(usage, field) + getattr(manager.usage, field))
    report.usage = usage.summary()  # the whole run, not just the Manager
    yield WarRoomEvent(kind="report", session_id=session_id, report=report, statuses=snapshot(), usage=manager.usage.summary())

    store.complete_session(session_id)
    yield WarRoomEvent(kind="done", session_id=session_id, statuses=snapshot(), usage=usage.summary())


def run_multi_agent(
    owner: str,
    repo: str,
    user_query: Optional[str] = None,
    *,
    store: Optional[MemoryStore] = None,
    client: Any = None,
    model: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
) -> WarRoomResult:
    """
    Execute the multi-agent War Room to completion.

    Specialists run in order, each saving structured findings to shared memory; the Manager
    then reads them all and produces the Repository Health Report.
    """
    events = list(stream_war_room(owner, repo, user_query, store=store, client=client, model=model, only=only))
    last = events[-1]
    report = next((e.report for e in events if e.kind == "report"), None)
    return WarRoomResult(
        session_id=next((e.session_id for e in events if e.session_id), ""),
        report=report,
        statuses=next((e.statuses for e in reversed(events) if e.statuses), []),
        events=events,
        usage=last.usage if last.kind == "done" else "",
        error=last.message if last.kind == "error" else None,
    )


def load_last_report(store: MemoryStore, repo_id: str) -> Optional[dict]:
    """The most recent finished War Room report for a repository (from memory), or None."""
    for session in store.list_sessions(repo_id, mode="multi_agent"):
        if not session.metadata:
            continue
        try:
            data = json.loads(session.metadata)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("report_markdown"):
            return {"session_id": session.session_id, "markdown": data["report_markdown"],
                    "narrative": data.get("narrative"), "generated_at": data.get("generated_at"), "statuses": [AgentStatus(**s) for s in data.get("statuses", [])]}
    return None
