"""
The Multi-Agent War Room screen.

Four tabs: run the team (live), read the report, compare with the single agent, inspect memory.
All logic lives in ``orchestration.runner``; this module only draws what it streams.
"""

from typing import Optional

import streamlit as st

from agents.multi import SPECIALISTS
from agents.multi.report import AgentStatus
from memory import MemoryStore
from orchestration.runner import load_last_report, stream_war_room
from ui.components import agent_status_dashboard, compare_answers, compare_view, get_session_usage, memory_panel, render_events, report_viewer


def _load_state(store: MemoryStore, repo_id: str) -> dict:
    """War Room state for a repository; picks up the last finished report from memory."""
    state = st.session_state.get("war_room")
    if state and state["repo_id"] == repo_id:
        return state
    saved = load_last_report(store, repo_id)
    state = {
        "repo_id": repo_id,
        "session_id": saved["session_id"] if saved else None,
        "statuses": saved["statuses"] if saved else [AgentStatus(agent_id=c.agent_id, title=c.title) for c in SPECIALISTS],
        "report": saved["markdown"] if saved else None,
        "narrative": saved["narrative"] if saved else None,
        "execution_note": saved["execution_note"] if saved else "",
        "usage": "",
    }
    st.session_state.war_room = state
    return state


def _run_team(store: MemoryStore, state: dict, owner: str, repo: str, chosen: list[str], focus: str, dashboard, live,
              max_output_tokens: int, lite_mode: bool, parallel: bool):
    """
    Stream one War Room run into the page.

    ``current`` is keyed by agent_id (plus "manager") rather than a single variable, because in
    parallel mode several specialists have a status widget open at the same time; a single shared
    variable would only ever point at the most recently started one, silently losing the others.
    """
    state.update(report=None, narrative=None, execution_note="", usage="")
    current: dict[str, "st.delta_generator.DeltaGenerator"] = {}
    for ev in stream_war_room(owner, repo, focus or None, store=store, only=chosen,
                               max_output_tokens=max_output_tokens, lite_mode=lite_mode, parallel=parallel):
        if ev.statuses:
            state["statuses"] = ev.statuses
            with dashboard.container():
                agent_status_dashboard(state["statuses"])
        if ev.kind == "error":
            st.error(ev.message)
        elif ev.kind == "start":
            state["session_id"] = ev.session_id
        elif ev.kind == "agent_start":
            current[ev.agent_id] = live.status(f"{ev.title} specialist working…", expanded=True)
        elif ev.kind == "agent" and ev.agent_id in current:
            with current[ev.agent_id]:
                render_events([ev.event], live=True)
        elif ev.kind == "agent_done" and ev.agent_id in current:
            status = next(s for s in ev.statuses if s.agent_id == ev.agent_id)
            current[ev.agent_id].update(label=f"{ev.title}: {status.findings} finding(s) · {ev.usage.split('·')[-1].strip()}",
                                         state="complete" if status.state == "done" else "error", expanded=False)
        elif ev.kind == "manager_start":
            current["manager"] = live.status("Manager writing the report…", expanded=False)
        elif ev.kind == "report":
            state["report"], state["narrative"], state["execution_note"] = ev.report.markdown, ev.report.narrative, ev.report.execution_note
            current["manager"].update(label="Manager: report ready", state="complete")
        elif ev.kind == "done":
            state["usage"] = ev.usage
            if ev.usage_meter is not None:
                get_session_usage().merge(ev.usage_meter)
            st.success("War Room finished. Open the **Report** tab.")


def render_war_room(store: MemoryStore, owner: str, repo: str, repo_id: str, max_output_tokens: int, lite_mode: bool,
                     identity: Optional[str] = None):
    # War Room reports are a shared team artifact by design (unlike the single-agent chat), so unlike
    # ``render_single_agent`` this intentionally does not scope resumption by ``identity``.
    state = _load_state(store, repo_id)
    st.caption(f"War Room for **{repo_id}**" + (" · showing the last saved report" if state["report"] else ""))
    run_tab, report_tab, compare_tab, memory_tab = st.tabs(["🛰️ War Room", "📋 Report", "🆚 Single vs Team", "🧠 Memory"])

    with run_tab:
        chosen = st.multiselect(
            "Specialists to run", [c.agent_id for c in SPECIALISTS], default=[c.agent_id for c in SPECIALISTS],
            format_func=lambda agent_id: next(c.title for c in SPECIALISTS if c.agent_id == agent_id),
        )
        focus = st.text_input("Optional focus for the team", placeholder="e.g. pay special attention to CI security")
        execution_mode = st.radio(
            "Execution mode", ["Sequential", "Parallel"], horizontal=True,
            help="Sequential (default): one specialist at a time, easier to follow; each sees what earlier "
                 "specialists already found. Parallel: all specialists run at once (faster), but cannot see "
                 "each other's findings while working.",
        )
        parallel = execution_mode == "Parallel"
        st.caption(
            ("Specialists run **at the same time**, then the Manager writes the report. " if parallel else
             "Specialists run **one after another**, then the Manager writes the report. ")
            + "Roughly 0.4–0.7¢ per specialist on gpt-4o-mini (less with LITE_MODE=1)."
        )
        run = st.button("🚀 Run War Room", type="primary", disabled=not chosen)
        dashboard = st.empty()
        live = st.container()
        if run:
            _run_team(store, state, owner, repo, chosen, focus.strip(), dashboard, live, max_output_tokens, lite_mode, parallel)
        else:
            with dashboard.container():
                agent_status_dashboard(state["statuses"])
        if state["usage"]:
            st.caption(f"💲 {state['usage']}")

    team = [f for f in store.get_findings(repo_id, session_id=state["session_id"]) if f.agent.endswith("_specialist")] if state["session_id"] else []

    with report_tab:  # drawn after the run so it shows the report that was just produced
        if state["report"]:
            st.subheader(f"Repository Health Report: {repo_id}")
            report_viewer(state["report"], narrative=state["narrative"], statuses=state["statuses"], findings=team,
                           execution_note=state["execution_note"])
            st.download_button("⬇️ Download report (Markdown)", state["report"], file_name=f"{repo_id.replace('/', '_')}_health_report.md",
                               mime="text/markdown")
        else:
            st.info("No report yet. Run the War Room in the first tab.")

    with compare_tab:
        single_sessions = store.list_sessions(repo_id, mode="single_agent", limit=1)
        single_answer = None
        if single_sessions:
            history = store.get_conversation_history(single_sessions[0].session_id)
            single_answer = next((m.content for m in reversed(history) if m.role == "assistant"), None)
        chat_state = st.session_state.get("chat") or {}
        single_usage = chat_state.get("usage") if chat_state.get("repo_id") == repo_id else None

        compare_answers(single_answer, single_usage, state["narrative"], state["usage"] or None)
        compare_view(store.get_findings(repo_id, agent="single_agent"), team)

    with memory_tab:
        memory_panel(store, repo_id)
