"""
System prompt and utilities for the single-agent investigation mode.

The single agent prompt emphasizes:
- Transparent, step-by-step investigation
- Deliberate tool choice justification
- Memory consultation before and after tool calls
- Clear stopping criteria when enough information is gathered
"""

from memory.schemas import ResumeContext

# One-click prompts offered by the UI; the agent handles them like any other message.
QUICK_PROMPTS = {
    "analyze": "Analyze this repository: what it does, how it is structured, its dependencies, and how healthy the project looks.",
    "teach": "Teach me this repo. Give me a structured explanation I can learn from.",
    "continue": "Continue from where we stopped last time. Summarize what you already know, then investigate what is still unexplored.",
}

MAX_EVIDENCE_CHARS = 200


def get_single_agent_system_prompt(max_steps: int = 10) -> str:
    """
    Get the system prompt for the single agent.

    The prompt guides the agent to:
    1. Start by understanding the repository context
    2. Deliberately choose which tools to call and why
    3. Consult memory for previous findings
    4. Update memory with new observations
    5. Answer the user's question when sufficient information is gathered

    Args:
        max_steps: Tool-calling rounds the agent is allowed before it must answer.
    """
    return f"""You are Agent 101, a careful, transparent software-repository investigator.
You investigate ONE public GitHub repository (given below) using the tools provided, and you remember what you learn.

## How to work
1. Look at the "Memory" section first. If it already answers the question, say so and answer from it; only use tools to fill gaps or verify.
2. Before EVERY tool call, write one short sentence saying what you are about to check and why. The user watches this.
3. Investigate deliberately, in this order unless the question says otherwise:
   README -> repository metadata -> file tree -> key source/config files -> dependency files -> recent activity (commits, issues, PRs, releases).
4. Prefer few, targeted calls over broad ones. Never re-fetch something you already have. You have at most {max_steps} tool-calling rounds; then you must answer.
5. Ground every claim in evidence you actually retrieved (file paths, numbers, dates). If you did not look, say you did not look. Never invent files or facts.
6. If a tool returns an error, adapt: try another path or tool, or report the limitation. If the error is a rate limit, stop investigating and tell the user how long to wait.

## Memory
- Call save_finding for each notable, durable observation (roughly 3-8 for a full analysis; none for a small factual question). Use severity "info" for neutral facts, "warning" for risks or smells, "critical" only for serious problems. Include concrete evidence.
- Do not save something already listed in Memory.
- When the user says things like "remember that I am learning X" or states a preference, call save_user_context (e.g. key "learning_focus"), then confirm briefly.
- For "what did you find about Y last time?" use Memory and recall_findings; do not re-investigate unless asked.
- For "is the <risk> you mentioned still present?": find the earlier finding, re-check the underlying evidence with tools, answer clearly (still present / resolved / cannot tell), and save a new finding recording the current status.
- If the user has a learning focus in Memory, tailor explanations to it.

## Answering
- Reply in concise Markdown. Lead with the answer.
- "Analyze this repository": cover purpose, architecture/structure, dependencies, project health, and notable risks.
- "Teach me this repo": use these sections: What it is; How it is organized; Key concepts; Where to start reading (specific files, in order); How to run or contribute.
- End by suggesting one or two useful follow-up questions."""


def build_memory_context(resume: ResumeContext) -> str:
    """Render everything remembered about a repository as a prompt section."""
    lines = [f"## Memory for {resume.repo_id}"]

    if resume.profile:
        p = resume.profile
        lines.append(
            f"Profile (last analyzed {p.last_analyzed:%Y-%m-%d}): {p.stars} stars, language {p.primary_language or 'unknown'}, "
            f"license {p.license or 'none'}, default branch {p.default_branch}. {p.description or ''}".rstrip()
        )
    if resume.last_session:
        lines.append(f"Last session: {resume.last_session.mode}, started {resume.last_session.created_at:%Y-%m-%d %H:%M} UTC.")

    if resume.user_context:
        lines.append("\nUser context (what the user told you to remember):")
        lines += [f"- {c.key}: {c.value}" for c in resume.user_context]

    if resume.findings:
        lines.append(f"\nPrevious findings (newest first, {len(resume.findings)} shown):")
        for f in resume.findings:
            evidence = f.evidence_json if f.evidence_json != "{}" else ""
            suffix = f" | evidence: {evidence[:MAX_EVIDENCE_CHARS]}" if evidence else ""
            lines.append(f"- #{f.id} [{f.created_at:%Y-%m-%d}] {f.severity}/{f.category} by {f.agent}: {f.finding}{suffix}")

    if len(lines) == 1:
        lines.append("Nothing yet: this repository has not been investigated before.")
    return "\n".join(lines)
