"""
Reusable Streamlit UI components for GitHub Agent Studio.

This module provides the building blocks shared by the single-agent and
War Room screens: repository and mode inputs, the repository card, tool-call
cards and timeline, the findings browser, the report viewer, and an error boundary.
"""

import csv
import io
import json
import logging
import traceback
from datetime import datetime
from typing import Callable, Optional

import streamlit as st

from agents.base import AgentEvent
from agents.multi.report import (
    DISCLAIMER_TEXT,
    confidence_label,
    count_by_severity,
    escape_markdown,
    evidence_text,
    sort_findings,
    unknown_citations,
)
from tools import parse_repo_ref
from tools.client import get_client
from ui.styles import demote_headings, format_markdown, severity_badge, severity_icon

MODES = ("single_agent", "multi_agent")
_MODE_LABELS = {"single_agent": "Single Agent (Agent 101)", "multi_agent": "Multi-Agent War Room"}
_STATE_ICON = {"waiting": "⏳", "running": "🔄", "done": "✅", "error": "❌", "skipped": "⏭️"}
MAX_FINDINGS_SHOWN = 100

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def repository_input(key: str = "repo_text"):
    """
    Display a repository input widget.

    Allows the user to enter owner/repo or a full GitHub URL, and validates the format.
    The text lives in ``st.session_state[key]``, so it survives a mode switch.

    Returns:
        Tuple of (owner, repo), or (None, None) if the input is empty or invalid
    """
    text = st.text_input("GitHub repository (owner/repo or URL)", placeholder="langchain-ai/langchain", key=key).strip()
    if not text:
        return None, None
    try:
        ref = parse_repo_ref(text)
    except ValueError:
        st.caption("⚠️ Use `owner/repo` or a github.com URL.")
        return None, None
    return ref.owner, ref.repo


def mode_selector(key: str = "mode") -> str:
    """
    Display the mode selector (Single Agent vs War Room).

    Returns:
        Selected mode string: "single_agent" or "multi_agent"
    """
    return st.radio(
        "Investigation mode", MODES, format_func=_MODE_LABELS.__getitem__, key=key,
        help="Single Agent: a transparent agent you can chat with. War Room: four specialists + a manager write a report.",
    )


# ---------------------------------------------------------------------------
# Repository card and API budget
# ---------------------------------------------------------------------------

def repo_header(info):
    """The repository card shown above both modes. ``info`` is a tools.schemas.RepositoryOutput."""
    title = f"[{info.owner}/{info.name}]({info.html_url})" if info.html_url else f"{info.owner}/{info.name}"
    st.markdown(f"### {title}")
    if info.description:
        st.caption(info.description)
    columns = st.columns(6)
    columns[0].metric("Stars", f"{info.stars:,}")
    columns[1].metric("Forks", f"{info.forks:,}")
    columns[2].metric("Language", info.language or "—")
    columns[3].metric("License", info.license or "—")
    columns[4].metric("Last push", (info.pushed_at or info.updated_at)[:10])
    columns[5].metric("Open issues + PRs", f"{info.open_issues_and_prs:,}")
    if info.is_archived:
        st.warning("This repository is archived, so it is read-only and probably not maintained.")


def github_rate_limit_caption():
    """Show the last-seen GitHub API budget, and warn when it is nearly gone."""
    core = get_client().rate_limit_status().get("core")
    if core is None:
        return
    resets = datetime.fromtimestamp(core.reset_at).strftime("%H:%M")
    st.caption(f"GitHub API: {core.remaining:,} requests left · resets {resets}")
    if core.remaining <= 5:
        st.warning("GitHub API budget almost used up. Add a GITHUB_TOKEN to .env for 5,000 requests/hour.")


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

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


def tool_timeline(events: list):
    """A compact table of every tool call in a run: which round, which tool, with what, and how it went."""
    rows = [
        {"Round": e.step, "Tool": e.name, "Arguments": json.dumps(e.arguments or {})[:70], "Result": e.content, "OK": "❌" if e.is_error else "✅"}
        for e in events if e.kind == "tool_result"
    ]
    if rows:
        st.dataframe(rows, hide_index=True)


def render_event(event: AgentEvent, live: bool = False):
    """Render one agent event. ``live`` also shows 'about to call' lines, which are redundant in a replay."""
    if event.kind == "reasoning":
        st.markdown("💭 " + escape_markdown(event.content, keep_emphasis=True))
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


# ---------------------------------------------------------------------------
# Findings browser
# ---------------------------------------------------------------------------

_SORTS = {
    "Most severe first": lambda fs: sort_findings(fs),
    "Newest first": lambda fs: sorted(fs, key=lambda f: f.id or 0, reverse=True),
    "Highest confidence": lambda fs: sorted(fs, key=lambda f: (-f.confidence, f.id or 0)),
}


def findings_csv(findings: list) -> str:
    """Findings as CSV text, for the export button."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "agent", "severity", "category", "confidence", "finding", "evidence", "created_at"])
    for f in findings:
        writer.writerow([f.id, f.agent, f.severity, f.category, f.confidence, f.finding, evidence_text(f), f.created_at.isoformat()])
    return out.getvalue()


def filter_findings(findings: list, *, agents=(), severities=(), categories=(), min_confidence: float = 0.0, search: str = "") -> list:
    """The filtering used by the findings browser (a pure function, so it is easy to test)."""
    needle = search.strip().lower()
    return [
        f for f in findings
        if (not agents or f.agent in agents)
        and (not severities or f.severity in severities)
        and (not categories or f.category in categories)
        and f.confidence >= min_confidence
        and (not needle or needle in f.finding.lower() or needle in evidence_text(f).lower())
    ]


def findings_inspector(findings: list, key: str = "findings"):
    """
    Browse findings: filter by agent, severity, category, confidence and text; sort; export to CSV.

    Args:
        findings: List of Finding objects
        key: Prefix for the widget keys (use a different one if two browsers share a page)
    """
    if not findings:
        st.info("No findings saved for this repository yet.")
        return
    a, b, c = st.columns(3)
    agents = a.multiselect("Agent", sorted({f.agent for f in findings}), key=f"{key}_agent")
    severities = b.multiselect("Severity", ["critical", "warning", "info"], key=f"{key}_severity")
    categories = c.multiselect("Category", sorted({f.category for f in findings}), key=f"{key}_category")
    d, e, f_ = st.columns([1, 2, 1])
    min_confidence = d.slider("Min confidence", 0.0, 1.0, 0.0, 0.1, key=f"{key}_conf")
    search = e.text_input("Search text or evidence", key=f"{key}_search")
    order = f_.selectbox("Sort", list(_SORTS), key=f"{key}_sort")

    shown = _SORTS[order](filter_findings(findings, agents=agents, severities=severities, categories=categories,
                                          min_confidence=min_confidence, search=search))
    left, right = st.columns([3, 1])
    left.caption(f"{len(shown)} of {len(findings)} findings")
    right.download_button("⬇️ Export CSV", findings_csv(shown), file_name="findings.csv", mime="text/csv", key=f"{key}_csv", disabled=not shown)
    if not shown:
        st.info("No findings match these filters.")
    for finding in shown[:MAX_FINDINGS_SHOWN]:
        with st.container(border=True):
            badge_col, text_col = st.columns([1, 6])
            with badge_col:
                severity_badge(finding.severity)
                st.caption(f"confidence {finding.confidence:.1f} ({confidence_label(finding.confidence)})")
            with text_col:
                st.markdown(f"**{escape_markdown(finding.category)}** · `{finding.agent}` · #{finding.id} · {finding.created_at:%Y-%m-%d}")
                st.markdown(escape_markdown(finding.finding))
                if evidence := evidence_text(finding):
                    st.caption("Evidence: " + escape_markdown(evidence))
    if len(shown) > MAX_FINDINGS_SHOWN:
        st.caption(f"Showing the first {MAX_FINDINGS_SHOWN}. Narrow the filters to see the rest.")


def sessions_table(store, repo_id: str):
    """Every investigation session for a repository: when, which mode, how many findings, finished or not."""
    sessions = store.list_sessions(repo_id)
    rows = [
        {
            "Started": f"{s.created_at:%Y-%m-%d %H:%M}",
            "Mode": "War Room" if s.mode == "multi_agent" else "Single agent",
            "Findings": len(store.get_findings(repo_id, session_id=s.session_id)),
            "Report": "✅" if s.metadata and "report_markdown" in s.metadata else "",
            "Finished": "✅" if s.completed_at else "",
        }
        for s in sessions
    ]
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("No sessions yet.")


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
    with st.expander("Sessions"):
        sessions_table(store, repo_id)


# ---------------------------------------------------------------------------
# War Room
# ---------------------------------------------------------------------------

def agent_status_dashboard(agents: list):
    """
    Display multi-agent status and progress: one card per team member.

    Args:
        agents: List of AgentStatus objects (waiting / running / done / error / skipped)
    """
    if not agents:
        return
    for column, agent in zip(st.columns(len(agents)), agents):
        with column:
            with st.container(border=True):
                st.markdown(f"**{agent.title}**")
                st.markdown(f"{_STATE_ICON[agent.state]} {agent.state}")
                if agent.state in ("done", "error"):
                    st.caption(f"{agent.findings} finding(s)")


def report_viewer(report: str, *, narrative: Optional[str] = None, statuses: Optional[list] = None, findings: Optional[list] = None):
    """
    Display a finished Repository Health Report.

    With the structured parts (narrative, statuses, findings) it draws a summary row, the Manager's
    narrative with its finding citations, and the evidence in one expander per specialist.
    Without them (an older saved report) it falls back to rendering the Markdown as it is.
    """
    if not (narrative and statuses is not None and findings is not None):
        st.markdown(format_markdown(demote_headings(report, 1)))
        return

    counts = count_by_severity(findings)
    columns = st.columns(5)
    columns[0].metric("Findings", len(findings))
    columns[1].metric("🔴 Critical", counts["critical"])
    columns[2].metric("🟠 Warning", counts["warning"])
    columns[3].metric("🔵 Info", counts["info"])
    columns[4].metric("Specialists done", f"{sum(s.state == 'done' for s in statuses)}/{len(statuses)}")
    st.info(f"**How to read this report.** {DISCLAIMER_TEXT}")
    for s in statuses:
        if s.state in ("error", "skipped"):
            st.warning(f"**{s.title}** {s.state}: {s.message or 'no details'}")
    if missing := unknown_citations(narrative, findings):
        st.warning("The narrative cites finding ids that do not exist: " + ", ".join(f"#{i}" for i in missing) + ". Check those claims against the evidence.")

    st.markdown(format_markdown(demote_headings(escape_markdown(narrative, keep_emphasis=True))))

    st.subheader("Evidence")
    st.caption("The `#id` tags in the text above match these findings.")
    for s in statuses:
        mine = sort_findings([f for f in findings if f.agent == s.agent_id])
        if not mine:
            continue
        with st.expander(f"{s.title} ({len(mine)})"):
            for f in mine:
                st.markdown(f"{severity_icon(f.severity)} **#{f.id}** `{f.severity}/{f.category}` · confidence {f.confidence:.1f} ({confidence_label(f.confidence)})  \n{escape_markdown(f.finding)}")
                if evidence := evidence_text(f):
                    st.caption("Evidence: " + escape_markdown(evidence))


def compare_view(single_findings: list, team_findings: list):
    """
    Side by side: what the single agent found versus what the War Room team found on the same repository.

    Args:
        single_findings: Findings saved by the single agent (any session)
        team_findings: Findings saved by the specialists in the latest War Room session
    """
    left, right = st.columns(2)
    for column, title, findings in ((left, "🤖 Single agent", single_findings), (right, "🛰️ War Room team", team_findings)):
        with column:
            st.subheader(title)
            st.caption(f"{len(findings)} finding(s)")
            if not findings:
                st.info("Nothing saved yet." if title.startswith("🤖") else "Run the War Room first.")
            for f in findings:
                st.markdown(f"{severity_icon(f.severity)} **{f.category}** · {f.agent.removesuffix('_specialist')} · conf {f.confidence:.1f}  \n{escape_markdown(f.finding)}")
    if single_findings and team_findings:
        overlap = sorted({f.category for f in single_findings} & {f.category for f in team_findings})
        st.caption("Categories both reported: " + (", ".join(overlap) if overlap else "none (they used different labels)"))


# ---------------------------------------------------------------------------
# Error boundary
# ---------------------------------------------------------------------------

def safe_render(render: Callable, *args, **kwargs):
    """
    Run a screen; if it fails, show a friendly message instead of a raw traceback.

    Streamlit's own control flow (``st.stop()``, ``st.rerun()``) is not an ``Exception``, so it still works.
    Saved memory is never touched by a rendering failure.
    """
    try:
        return render(*args, **kwargs)
    except Exception as exc:
        logger.exception("UI error in %s", getattr(render, "__name__", render))
        st.error(f"Something went wrong ({type(exc).__name__}). Your saved memory is safe. Try again, or reload the page.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
