"""
Shared base for the four War Room specialists.

A specialist is the same tool-calling loop as the single agent, with three differences:
  - it gets a focused system prompt and only the tools its job needs
  - its real output is *structured findings* saved to shared memory (never a free-form report)
  - it sees what earlier specialists already recorded in this session

Subclasses only declare who they are, which tools they may use, and their prompt.
"""

from typing import Any, Callable, ClassVar, Iterator, List, Optional

from agents.base import Agent, AgentEvent, InvestigationResult
from agents.loop import resolve_llm, run_tool_loop
from agents.toolbox import Toolbox
from memory import Finding, MemoryStore
from prompts.multi_agent import build_shared_context

DEFAULT_SPECIALIST_MAX_STEPS = 8
# A specialist's turns are short: a sentence plus a few save_finding calls (~100 tokens each, several per round).
DEFAULT_SPECIALIST_MAX_OUTPUT_TOKENS = 1536
MAX_ATTEMPTS = 2  # the second attempt is a nudge for specialists that finish without saving anything

_NUDGE = (
    "You finished without recording any findings, and the Manager can only see saved findings. "
    "Record what you found now with save_finding (at least one; if nothing was notable or your tools were blocked, "
    "record that as an 'info' or 'warning' finding), then reply with a short closing note."
)

# Every specialist may record findings and read what teammates recorded.
_MEMORY_TOOLS = ("save_finding", "recall_findings")


class SpecialistResult(InvestigationResult):
    findings: List[Finding] = []  # what this specialist saved to shared memory


class SpecialistAgent(Agent):
    """Base class: subclasses set ``agent_id``, ``title``, ``tools`` and ``prompt``."""

    agent_id: ClassVar[str]
    title: ClassVar[str]  # display name, e.g. "Architecture"
    tools: ClassVar[tuple[str, ...]]  # GitHub tools this specialist may call
    prompt: ClassVar[Callable[[int], str]]  # max_steps -> system prompt

    def __init__(
        self,
        store: MemoryStore,
        client: Any = None,
        model: Optional[str] = None,
        max_steps: int = DEFAULT_SPECIALIST_MAX_STEPS,
        max_output_tokens: int = DEFAULT_SPECIALIST_MAX_OUTPUT_TOKENS,
    ):
        super().__init__(self.agent_id)
        self.store, self.max_steps, self.max_output_tokens = store, max_steps, max_output_tokens
        self._client, self._model = client, model

    def start_session(self, owner: str, repo: str) -> str:
        return self.store.create_session(MemoryStore.make_repo_id(owner, repo), "multi_agent")

    def investigate(self, owner: str, repo: str, query: Optional[str] = None, session_id: Optional[str] = None) -> SpecialistResult:
        """Run to completion. Findings are in shared memory and echoed on the result."""
        session_id = session_id or self.start_session(owner, repo)
        events = list(self.run(owner, repo, session_id, query))
        last = events[-1]
        return SpecialistResult(
            session_id=session_id,
            answer=last.content if last.kind == "final" else "",
            events=events,
            steps=max((e.step for e in events), default=0),
            error=last.content if last.kind == "error" else None,
            findings=self._my_findings(MemoryStore.make_repo_id(owner, repo), session_id),
        )

    def run(self, owner: str, repo: str, session_id: str, query: Optional[str] = None) -> Iterator[AgentEvent]:
        """Yield events as the specialist works; the last one is 'final' or 'error'."""
        try:
            client, model = resolve_llm(self._client, self._model)
        except Exception as exc:
            yield AgentEvent(kind="error", is_error=True, content=f"Could not start the OpenAI client: {exc}")
            return

        repo_id = MemoryStore.make_repo_id(owner, repo)
        toolbox = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, include=(*self.tools, *_MEMORY_TOOLS))
        teammates = [f for f in self.store.get_findings(repo_id, session_id=session_id) if f.agent != self.agent_id]
        task = f"Audit {owner}/{repo} from your specialist perspective and record your structured findings."
        if query:
            task += f" Extra focus from the user: {query}"
        messages: list[dict] = [
            {"role": "system", "content": self.prompt(self.max_steps) + "\n\n" + build_shared_context(teammates[::-1])},
            {"role": "user", "content": task},
        ]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = None
            for event in run_tool_loop(client, model, messages, toolbox, max_steps=self.max_steps, max_output_tokens=self.max_output_tokens):
                if event.kind == "final":
                    final = event
                    break
                yield event
                if event.kind == "error":
                    return
            if final is None:
                return
            if attempt == MAX_ATTEMPTS or self._my_findings(repo_id, session_id):
                yield final
                return
            yield AgentEvent(kind="reasoning", step=final.step, content="Finished without saving any findings; asking for them.")
            messages.append({"role": "user", "content": _NUDGE})

    def _my_findings(self, repo_id: str, session_id: str) -> List[Finding]:
        return self.store.get_findings(repo_id, agent=self.agent_id, session_id=session_id)
