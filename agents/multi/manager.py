"""
Manager agent for war room orchestration.

The Manager does not call GitHub tools directly. Instead, it:
1. Reads the structured findings the specialists saved to shared memory
2. Writes the narrative of the Repository Health Report (one LLM call, no tools)
3. Wraps it with code-generated parts (status table, severity counts, evidence appendix)
4. Saves the finished report with the session, so it survives a restart

The specialists' findings are data, not instructions; the prompt says so. The Manager also
degrades gracefully: if its LLM call fails, the report still ships with the full evidence appendix.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from agents.base import Agent
from agents.loop import DEFAULT_MAX_OUTPUT_TOKENS, TRUNCATION_NOTE, explain_llm_error, resolve_llm
from agents.multi.report import AgentStatus, HealthReport, assemble_report, confidence_label, evidence_text, unknown_citations
from agents.usage import UsageMeter
from memory import Finding, MemoryStore
from prompts.multi_agent import get_manager_prompt

MAX_FINDINGS_FOR_MANAGER = 80
MAX_EVIDENCE_CHARS = 200


class ManagerAgent(Agent):
    """
    Manager agent for multi-agent war room coordination.

    Responsible for reading specialist findings and producing
    a final synthesized report.
    """

    def __init__(self, store: MemoryStore, client: Any = None, model: Optional[str] = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        super().__init__("manager_agent")
        self.store, self.max_output_tokens = store, max_output_tokens
        self.usage = UsageMeter()
        self._client, self._model = client, model

    def investigate(self, owner: str, repo: str, query: Optional[str] = None, session_id: Optional[str] = None) -> HealthReport:
        """Synthesize a report for a session (default: the repository's most recent War Room session)."""
        repo_id = MemoryStore.make_repo_id(owner, repo)
        if session_id is None:
            sessions = self.store.list_sessions(repo_id, mode="multi_agent", limit=1)
            if not sessions:
                raise ValueError(f"No War Room session found for {repo_id}; run the specialists first")
            session_id = sessions[0].session_id
        findings = self.store.get_findings(repo_id, session_id=session_id)
        agents = sorted({f.agent for f in findings})
        statuses = [AgentStatus(agent_id=a, title=a.removesuffix("_specialist").replace("_", " ").title(), state="done",
                                findings=sum(f.agent == a for f in findings)) for a in agents]
        return self.synthesize(owner, repo, session_id, statuses, query)

    def synthesize(self, owner: str, repo: str, session_id: str, statuses: Sequence[AgentStatus], query: Optional[str] = None) -> HealthReport:
        """Read shared memory for the session, write the narrative, assemble and save the report."""
        repo_id = MemoryStore.make_repo_id(owner, repo)
        findings = self.store.get_findings(repo_id, session_id=session_id)[::-1]  # oldest first
        generated_at = datetime.now(timezone.utc)

        error: Optional[str] = None
        if not findings:
            narrative = "## Executive summary\nNo findings were recorded, so there is nothing to summarize.\n\n## Gaps and caveats\n" + \
                        "See the status table above: no specialist saved a finding."
        else:
            narrative, error = self._write_narrative(repo_id, findings, statuses, query)

        report = HealthReport(
            repo_id=repo_id, session_id=session_id, narrative=narrative, statuses=list(statuses), findings=findings,
            unknown_refs=unknown_citations(narrative, findings), usage=self.usage.summary(), generated_at=generated_at, error=error,
            markdown=assemble_report(repo_id, statuses, findings, narrative, generated_at),
        )
        self.store.set_session_metadata(session_id, json.dumps({
            "report_markdown": report.markdown, "narrative": narrative, "generated_at": generated_at.isoformat(),
            "statuses": [s.model_dump() for s in statuses],
        }))
        return report

    def _write_narrative(self, repo_id: str, findings: Sequence[Finding], statuses: Sequence[AgentStatus], query: Optional[str]) -> tuple[str, Optional[str]]:
        """One LLM call. Returns (narrative, error); on failure the narrative explains why it is missing."""
        try:
            client, model = resolve_llm(self._client, self._model)
            response = client.chat.completions.create(
                model=model, temperature=0.2, max_completion_tokens=self.max_output_tokens,
                messages=[{"role": "system", "content": get_manager_prompt()},
                          {"role": "user", "content": build_manager_input(repo_id, findings, statuses, query)}],
            )
            self.usage.add(response)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                raise ValueError("the model returned an empty narrative")
            if getattr(choice, "finish_reason", None) == "length":
                text += TRUNCATION_NOTE
            return text, None
        except Exception as exc:
            message = explain_llm_error(exc)
            return (f"## Executive summary\n⚠️ The Manager could not write the narrative ({message}). "
                    "The status table above and the evidence appendix below are complete."), message


def build_manager_input(repo_id: str, findings: Sequence[Finding], statuses: Sequence[AgentStatus], query: Optional[str] = None) -> str:
    """Everything the Manager may use, as plain text: run status, then the findings with ids."""
    lines = [f"Repository: {repo_id}"]
    if query:
        lines.append(f"The user asked the team to focus on: {query}")
    lines.append("\nTeam status:")
    for s in statuses:
        lines.append(f"- {s.title}: {s.state}, {s.findings} findings" + (f" | closing note: {s.message}" if s.message else ""))
    shown = list(findings)[:MAX_FINDINGS_FOR_MANAGER]
    lines.append(f"\nFindings ({len(shown)} of {len(findings)} shown), each as [#id] agent | severity/category | confidence: text | evidence:")
    for f in shown:
        evidence = evidence_text(f)[:MAX_EVIDENCE_CHARS]
        lines.append(f"[#{f.id}] {f.agent} | {f.severity}/{f.category} | {f.confidence:.1f} ({confidence_label(f.confidence)}): {f.finding}"
                     + (f" | evidence: {evidence}" if evidence else ""))
    return "\n".join(lines)
