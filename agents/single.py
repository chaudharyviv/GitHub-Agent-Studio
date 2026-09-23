"""
Single Agent (Agent 101) implementation.

The single agent is the educational heart of the system. It demonstrates
a transparent tool-calling loop using OpenAI's gpt-4o-mini model.

What this file adds around the shared loop (``agents/loop.py``):
  1. build the messages: system prompt + memory, earlier chat, the new question
  2. run the loop, yielding every step as an ``AgentEvent`` so a UI can show it live
  3. remember the conversation, so follow-up questions have context
  4. after a real investigation that saved nothing, a silent nudge asking it to save what it found

Findings and user context are saved by the model itself, through memory tools; the prompt asks for
this but (unlike specialists) nothing enforces it, hence step 4.
"""

from typing import Any, Iterator, Optional

from agents.base import Agent, AgentEvent, InvestigationResult, final_or_error
from agents.limits import Limits, get_limits
from agents.llm import LLMProvider
from agents.loop import DEFAULT_MAX_OUTPUT_TOKENS, resolve_llm, run_tool_loop, tool_log_entry
from agents.toolbox import Toolbox
from agents.usage import UsageMeter
from memory import ConversationMessage, MemoryStore
from prompts.single_agent import build_memory_context, get_single_agent_system_prompt

_MEMORY_TOOLS = ("save_finding", "recall_findings")
_NUDGE_STEPS = 2
# Below this many investigative (non-memory) tool calls, a turn is small/conversational and a nudge would
# just be annoying (or waste a call) if it saved nothing on purpose; at or above it, a real investigation
# that saved nothing looks like the prompt was simply not followed, worth one silent, cheap follow-up.
_INVESTIGATIVE_TOOL_THRESHOLD = 2
_SAVE_NUDGE = (
    "Before this turn ends: if that investigation surfaced any notable, durable finding, record it now with "
    "save_finding (skip if there is truly nothing worth remembering). Your answer above already stands; say nothing more."
)


class SingleAgent(Agent):
    """
    Single transparent agent for repository investigation.

    Uses OpenAI gpt-4o-mini for tool-calling decisions with explicit
    reasoning and memory updates visible throughout the process.
    """

    def __init__(self, store: MemoryStore, client: Any = None, model: Optional[str] = None, max_steps: Optional[int] = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, limits: Optional[Limits] = None, identity: Optional[str] = None,
                 provider: Optional[LLMProvider] = None):
        """
        Args:
            store: Long-term memory (findings, user context, chat history)
            client: OpenAI client; created from OPENAI_API_KEY when omitted
            model: Model name; defaults to OPENAI_MODEL (gpt-4o-mini)
            max_steps: Most tool-calling rounds before the agent must answer (default 10, or 6 in LITE_MODE)
            max_output_tokens: Cap on tokens the model may generate per call
            limits: Size limits (tool result caps, etc); defaults to ``LITE_MODE``-derived limits when omitted
            identity: Optional user-chosen name, so this session and its conversation don't get pulled
                into another named person's chat on a shared instance (see ``MemoryStore.list_sessions``).
                None (the default) means shared/anonymous, same as before this existed.
            provider: An ``LLMProvider`` to use directly instead of building one from ``client``/``model``
                (the extension point for a non-OpenAI provider); ``client``/``model`` are ignored when given.
        """
        super().__init__("single_agent")
        self.store, self.max_output_tokens, self.identity = store, max_output_tokens, identity
        self.limits = limits or get_limits()
        self.max_steps = max_steps or self.limits.single_agent_steps
        self.usage = UsageMeter()  # tokens and estimated cost across everything this agent has run
        self._client, self._model, self._provider = client, model, provider

    def start_session(self, owner: str, repo: str) -> str:
        return self.store.create_session(MemoryStore.make_repo_id(owner, repo), "single_agent", identity=self.identity)

    def investigate(self, owner: str, repo: str, query: Optional[str] = None, session_id: Optional[str] = None) -> InvestigationResult:
        """Run to completion and return everything (use ``run`` to watch it live)."""
        session_id = session_id or self.start_session(owner, repo)
        events = list(self.run(owner, repo, query or "Analyze this repository.", session_id))
        outcome = final_or_error(events)
        return InvestigationResult(
            session_id=session_id,
            answer=outcome.content if outcome and outcome.kind == "final" else "",
            events=events,
            steps=max((e.step for e in events), default=0),
            error=outcome.content if outcome and outcome.kind == "error" else None,
        )

    def run(self, owner: str, repo: str, query: str, session_id: str) -> Iterator[AgentEvent]:
        """
        The agent loop. Yields events as they happen; use ``agents.base.final_or_error`` to find the turn's
        outcome rather than assuming it is the last event — a silent save-finding nudge (see below) can add
        a few more events after it.
        """
        try:
            provider = resolve_llm(self._client, self._model, self._provider)
        except Exception as exc:
            yield AgentEvent(kind="error", is_error=True, content=f"Could not start the OpenAI client: {exc}")
            return

        repo_id = MemoryStore.make_repo_id(owner, repo)
        toolbox = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, limits=self.limits)
        system = get_single_agent_system_prompt(self.max_steps) + "\n\n" + build_memory_context(
            self.store.get_resume_context(repo_id, identity=self.identity))
        history = self.store.get_conversation_history(session_id, limit=self.limits.history_limit)
        messages: list[dict] = [{"role": "system", "content": system}]
        messages += [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": query})
        self.store.save_conversation_message(ConversationMessage(session_id=session_id, role="user", content=query))

        tool_log: list[str] = []
        investigated = 0  # non-memory tool calls this turn, for the save-finding nudge below
        saved_finding = False
        final_event: Optional[AgentEvent] = None
        for event in run_tool_loop(provider, messages, toolbox, max_steps=self.max_steps,
                                 max_output_tokens=self.max_output_tokens, usage=self.usage):
            if event.kind == "tool_result":
                tool_log.append(tool_log_entry(event))
                if event.name == "save_finding":
                    saved_finding = saved_finding or bool((event.data or {}).get("saved"))
                elif not event.is_error and event.name not in (*_MEMORY_TOOLS, "save_user_context"):
                    investigated += 1
            elif event.kind == "final":
                final_event = event
            yield event

        # The prompt asks the model to save findings (see prompts/single_agent.py), but nothing enforces
        # it, unlike specialists (which get a step-budget reminder and a retry). A real investigation that
        # saved nothing would otherwise leave no trace for later recall, so nudge it once, silently: the
        # visible answer above already stands either way; this only affects what memory remembers.
        if final_event is not None and not saved_finding and investigated >= _INVESTIGATIVE_TOOL_THRESHOLD:
            messages.append({"role": "system", "content": _SAVE_NUDGE})
            nudge_box = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, include=_MEMORY_TOOLS, limits=self.limits)
            for event in run_tool_loop(provider, messages, nudge_box, max_steps=_NUDGE_STEPS,
                                        max_output_tokens=self.max_output_tokens, usage=self.usage, last_round_tools=_MEMORY_TOOLS):
                if event.kind == "final":
                    break  # this round's own closing text is discarded; the turn's real answer already stands
                if event.kind == "error":
                    yield AgentEvent(kind="memory", content=f"Could not save additional findings automatically ({event.content}).")
                    break
                if event.kind == "tool_result":
                    tool_log.append(tool_log_entry(event))
                yield event

        if final_event is not None:
            self.store.save_conversation_message(
                ConversationMessage(session_id=session_id, role="assistant", content=final_event.content, tool_calls=tool_log or None)
            )
