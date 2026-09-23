"""
The parts of the Repository Health Report that are built by code, not by the LLM.

The Manager only writes the narrative. Everything factual is assembled here straight from
shared memory, so it cannot be misquoted: the status table, the severity counts, and the
evidence appendix listing every finding with its confidence and evidence.
"""

import json
import re
from datetime import datetime, timezone
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from memory.schemas import Finding

AgentState = Literal["waiting", "running", "done", "error", "skipped"]

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
_STATE_ICON = {"waiting": "⏳", "running": "🔄", "done": "✅", "error": "❌", "skipped": "⏭️"}

DISCLAIMER_TEXT = (
    "It was produced by AI agents doing automated, *static* analysis of public GitHub data (no code was run). "
    "Findings are leads to verify, not verified facts. Each carries a confidence score and its evidence; "
    "treat low-confidence items as hypotheses."
)
DISCLAIMER = f"> **How to read this report.** {DISCLAIMER_TEXT}"

LIMITATIONS_TEXT = (
    "## Limitations of this analysis\n"
    "- Public repositories only.\n"
    "- Security findings combine dependency-file inspection with an optional live CVE lookup (Tavily search, "
    "biased toward NVD / GitHub Advisories). This is still **not** a real vulnerability scan, SCA pipeline, or "
    "exploit verification — a returned CVE id is a lead to verify, not a confirmed match.\n"
    "- No full static analysis, license-compliance engine, or secret scanning.\n"
    "- Analysis depth is limited by step budgets, result-size caps, and model context.\n"
    "- Results depend on the capabilities of the configured model (`gpt-4o-mini` by default) and the system prompts."
)


class AgentStatus(BaseModel):
    """How one team member's run went."""
    agent_id: str
    title: str
    state: AgentState = "waiting"
    findings: int = 0
    message: str = ""  # closing note, or the error


class HealthReport(BaseModel):
    repo_id: str
    session_id: str
    markdown: str  # the complete report: header + narrative + appendix
    narrative: str  # the Manager's part only
    execution_note: str = ""  # code-generated "How the team ran" section (also folded into `markdown`)
    statuses: list[AgentStatus] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    unknown_refs: list[int] = Field(default_factory=list)  # finding ids the narrative cites that do not exist
    usage: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None  # set if the Manager could not write the narrative


def escape_markdown(text: str, *, keep_emphasis: bool = False) -> str:
    """
    Make model-written text safe to show as Markdown.

    Without this, a finding that mentions ``src/click/__init__.py`` renders as "src/click/**init**.py".
    Text inside `backtick code spans` is left alone. With ``keep_emphasis`` only underscores are escaped,
    so intended **bold** and *italic* survive (used for the Manager's narrative).
    """
    special = "_" if keep_emphasis else "_*~<[]"
    pattern = re.compile("([" + re.escape(special) + "])")
    parts = re.split(r"(`[^`\n]*`)", text)  # odd items are code spans
    return "".join(part if i % 2 else pattern.sub(r"\\\1", part) for i, part in enumerate(parts))


def evidence_text(finding: Finding) -> str:
    try:
        data = json.loads(finding.evidence_json)
    except ValueError:
        return ""
    return str(data.get("evidence", "")) if isinstance(data, dict) else ""


def confidence_label(confidence: float) -> str:
    return "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"


def sort_findings(findings: Sequence[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.id or 0))


def count_by_severity(findings: Sequence[Finding]) -> dict[str, int]:
    return {sev: sum(f.severity == sev for f in findings) for sev in _SEVERITY_ORDER}


def cited_ids(narrative: str) -> set[int]:
    """Finding ids the narrative cites, written as [#12]."""
    return {int(n) for n in re.findall(r"\[#(\d+)\]", narrative)}


def unknown_citations(narrative: str, findings: Sequence[Finding]) -> list[int]:
    return sorted(cited_ids(narrative) - {f.id for f in findings})


def render_header(repo_id: str, statuses: Sequence[AgentStatus], findings: Sequence[Finding], generated_at: datetime) -> str:
    counts = count_by_severity(findings)
    lines = [
        f"# Repository Health Report: {repo_id}",
        f"_Generated {generated_at:%Y-%m-%d %H:%M} UTC · {len(findings)} findings from {sum(s.state == 'done' for s in statuses)} of {len(statuses)} specialists · "
        f"🔴 {counts['critical']} critical · 🟠 {counts['warning']} warning · 🔵 {counts['info']} info_",
        "",
        DISCLAIMER,
        "",
        "| Specialist | Status | Findings |",
        "|---|---|---|",
    ]
    for s in statuses:
        note = f" ({s.message})" if s.state in ("error", "skipped") and s.message else ""
        lines.append(f"| {s.title} | {_STATE_ICON[s.state]} {s.state}{note} | {s.findings} |")
    return "\n".join(lines)


def render_appendix(findings: Sequence[Finding], statuses: Sequence[AgentStatus]) -> str:
    """Every finding, grouped by specialist, most severe first, with confidence and evidence."""
    lines = ["## Evidence: all findings", "Ids match the `[#id]` citations above."]
    for status in statuses:
        mine = sort_findings([f for f in findings if f.agent == status.agent_id])
        if not mine:
            continue
        lines += ["", f"### {status.title} ({len(mine)})"]
        for f in mine:
            lines.append(f"- {_SEVERITY_ICON[f.severity]} **[#{f.id}]** `{f.severity}/{f.category}` · confidence {f.confidence:.1f} ({confidence_label(f.confidence)}): {escape_markdown(f.finding)}")
            if evidence := evidence_text(f):
                lines.append(f"  - _Evidence:_ {escape_markdown(evidence)}")
    return "\n".join(lines)


def execution_note_text(parallel: bool) -> str:
    """Code-generated 'how the team ran' section: the isolation trade-off is a fact of the run, not the Manager's opinion."""
    if parallel:
        return (
            "## How the team ran\n"
            "Specialists ran **in parallel** for speed. They did not see each other's findings during execution; "
            "the Manager is the only component that sees the complete picture."
        )
    return (
        "## How the team ran\n"
        "Specialists ran **sequentially**, each one able to see what earlier specialists had already found."
    )


def assemble_report(
    repo_id: str, statuses: Sequence[AgentStatus], findings: Sequence[Finding], narrative: str, generated_at: datetime,
    parallel: bool = False,
) -> str:
    """Header + the Manager's narrative + appendix, with a warning if the narrative cites ids that do not exist."""
    parts = [render_header(repo_id, statuses, findings, generated_at)]
    if missing := unknown_citations(narrative, findings):
        parts.append("> ⚠️ The narrative below cites finding ids that do not exist: " + ", ".join(f"#{i}" for i in missing) + ". Check those claims against the appendix.")
    parts += [narrative.strip(), render_appendix(findings, statuses), execution_note_text(parallel), LIMITATIONS_TEXT]
    return "\n\n".join(parts) + "\n"
