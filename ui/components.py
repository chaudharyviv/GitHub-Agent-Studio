"""
Reusable Streamlit UI components for GitHub Agent Studio.

This module provides the building blocks for both single-agent and
multi-agent investigation interfaces.
"""

import streamlit as st

from agents.base import AgentEvent

_SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


def repository_input():
    """
    Display a repository input widget.
    
    Allows user to enter owner/repo or full GitHub URL.
    Validates format before returning.
    
    Returns:
        Tuple of (owner, repo) or (None, None) if invalid input
        
    Raises:
        NotImplementedError: Pending implementation in Phase 6
    """
    raise NotImplementedError("Repository input widget will be implemented in Phase 6: UI Polish")


def mode_selector():
    """
    Display mode selector (Single Agent vs War Room).
    
    Returns:
        Selected mode string: "single_agent" or "multi_agent"
        
    Raises:
        NotImplementedError: Pending implementation in Phase 6
    """
    raise NotImplementedError("Mode selector widget will be implemented in Phase 6: UI Polish")


def tool_call_card(tool_name: str, input_data: dict, output_data: dict, summary: str = "", is_error: bool = False):
    """
    Display a single tool call as an expandable card.

    Shows tool name, inputs, and outputs.

    Args:
        tool_name: Name of the tool called
        input_data: Input parameters
        output_data: Tool output
        summary: One-line result summary shown in the card title
        is_error: Whether the tool returned an error
    """
    icon = "❌" if is_error else "🔧"
    with st.expander(f"{icon} {tool_name} — {summary}" if summary else f"{icon} {tool_name}"):
        st.caption("Arguments")
        st.json(input_data or {}, expanded=False)
        st.caption("Result")
        st.json(output_data, expanded=False)


def render_event(event: AgentEvent, live: bool = False):
    """Render one agent event. ``live`` also shows 'about to call' lines, which are redundant in a replay."""
    if event.kind == "reasoning":
        st.markdown(f"💭 {event.content}")
    elif event.kind == "tool_call":
        if live:
            st.caption(f"⏳ calling `{event.name}` {event.content}")
    elif event.kind == "tool_result":
        tool_call_card(event.name or "tool", event.arguments or {}, event.data, event.content, event.is_error)
    elif event.kind == "memory":
        st.caption(f"🧠 {event.content}")
    elif event.kind == "error":
        st.error(event.content)


def render_events(events: list, live: bool = False):
    for event in events:
        if event.kind != "final":
            render_event(event, live)


def findings_inspector(findings: list):
    """
    Display findings with filtering.

    Args:
        findings: List of Finding objects
    """
    if not findings:
        st.info("No findings saved for this repository yet.")
        return
    col_agent, col_severity = st.columns(2)
    agents = col_agent.multiselect("Agent", sorted({f.agent for f in findings}))
    severities = col_severity.multiselect("Severity", ["critical", "warning", "info"])
    shown = [f for f in findings if (not agents or f.agent in agents) and (not severities or f.severity in severities)]
    st.caption(f"{len(shown)} of {len(findings)} findings")
    for f in shown:
        st.markdown(f"{_SEVERITY_ICON[f.severity]} **{f.category}** · {f.agent} · {f.created_at:%Y-%m-%d}  \n{f.finding}")
        if f.evidence_json != "{}":
            st.caption(f"Evidence: {f.evidence_json}")


def memory_panel(store, repo_id: str):
    """Everything remembered about a repository: profile, user context, sessions and findings."""
    resume = store.get_resume_context(repo_id, max_findings=500)
    if resume.profile:
        p = resume.profile
        st.subheader("Repository profile")
        cols = st.columns(4)
        cols[0].metric("Stars", f"{p.stars:,}")
        cols[1].metric("Language", p.primary_language or "—")
        cols[2].metric("License", p.license or "—")
        cols[3].metric("Last analyzed", f"{p.last_analyzed:%Y-%m-%d}")
    st.subheader("What you asked me to remember")
    if resume.user_context:
        for c in resume.user_context:
            st.markdown(f"- **{c.key}**: {c.value}")
    else:
        st.caption("Nothing yet. Try “Remember that I am learning the authentication module”.")
    st.subheader("Findings")
    findings_inspector(resume.findings)
    sessions = store.list_sessions(repo_id)
    st.caption(f"{len(sessions)} session(s) stored for this repository.")


def agent_status_dashboard(agents: list):
    """
    Display multi-agent status and progress.
    
    Shows which agents are running, completed, or errored.
    
    Args:
        agents: List of agent status objects
        
    Raises:
        NotImplementedError: Pending implementation in Phase 6
    """
    raise NotImplementedError("Agent status dashboard will be implemented in Phase 6: UI Polish")


def report_viewer(report: str):
    """
    Display a formatted investigation report.
    
    Renders Markdown report with syntax highlighting and structure.
    
    Args:
        report: Markdown report content
        
    Raises:
        NotImplementedError: Pending implementation in Phase 6
    """
    raise NotImplementedError("Report viewer will be implemented in Phase 6: UI Polish")
