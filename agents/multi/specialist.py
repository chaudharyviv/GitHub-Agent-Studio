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
from agents.limits import Limits, get_limits
from agents.llm import LLMProvider
from agents.loop import resolve_llm, run_tool_loop
from agents.toolbox import Toolbox
from agents.usage import UsageMeter
from memory import Finding, MemoryStore
from prompts.multi_agent import PROSE_REMINDER, SAVE_REMINDER, build_shared_context

# A specialist's turns are short: a sentence plus a few save_finding calls (~100 tokens each, several per round).
DEFAULT_SPECIALIST_MAX_OUTPUT_TOKENS = 1536
MAX_ATTEMPTS = 2  # the second attempt is a nudge for specialists that finish without saving anything
NUDGE_STEPS = 2  # the nudge attempt only offers the memory tools, so it needs very few rounds

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
    categories: ClassVar[tuple[str, ...]]  # finding categories save_finding accepts from it
    prompt: ClassVar[Callable[[int], str]]  # max_steps -> system prompt

    def __init__(
        self,
        store: MemoryStore,
        client: Any = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_output_tokens: int = DEFAULT_SPECIALIST_MAX_OUTPUT_TOKENS,
        limits: Optional[Limits] = None,
        provider: Optional[LLMProvider] = None,
    ):
        super().__init__(self.agent_id)
        self.store, self.max_output_tokens = store, max_output_tokens
        self.limits = limits or get_limits()
        self.max_steps = max_steps or self.limits.specialist_steps  # 8, or 5 in LITE_MODE
        self.usage = UsageMeter()  # tokens and estimated cost across everything this agent has run
        self._client, self._model, self._provider = client, model, provider

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
            provider = resolve_llm(self._client, self._model, self._provider)
        except Exception as exc:
            yield AgentEvent(kind="error", is_error=True, content=f"Could not start the OpenAI client: {exc}")
            return

        repo_id = MemoryStore.make_repo_id(owner, repo)
        toolbox = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, include=(*self.tools, *_MEMORY_TOOLS), categories=self.categories, limits=self.limits)
        teammates = [f for f in self.store.get_findings(repo_id, session_id=session_id) if f.agent != self.agent_id]
        task = f"Audit {owner}/{repo} from your specialist perspective and record your structured findings."
        if query:
            task += f" Extra focus from the user: {query}"
        messages: list[dict] = [
            {"role": "system", "content": self.prompt(self.max_steps) + "\n\n" + build_shared_context(teammates[::-1])},
            {"role": "user", "content": task},
        ]

        active_box, active_steps = toolbox, self.max_steps
        for attempt in range(1, MAX_ATTEMPTS + 1):
            final = None
            for event in run_tool_loop(provider, messages, active_box, max_steps=active_steps,
                                       max_output_tokens=self.max_output_tokens, remind_when_left=2 if attempt == 1 else 0, reminder=SAVE_REMINDER,
                                       usage=self.usage, last_round_tools=_MEMORY_TOOLS,
                                       max_reasoning_chars=self.limits.max_reasoning_chars, prose_reminder=PROSE_REMINDER):
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
            # The nudge can only save: no more exploring, so it is short and cheap.
            active_box = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, include=_MEMORY_TOOLS, categories=self.categories, limits=self.limits)
            active_steps = NUDGE_STEPS

    def _my_findings(self, repo_id: str, session_id: str) -> List[Finding]:
        return self.store.get_findings(repo_id, agent=self.agent_id, session_id=session_id)
