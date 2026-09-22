"""
Single Agent (Agent 101) implementation.

The single agent is the educational heart of the system. It demonstrates
a transparent tool-calling loop using OpenAI's gpt-4o-mini model.

What this file adds around the shared loop (``agents/loop.py``):
  1. build the messages: system prompt + memory, earlier chat, the new question
  2. run the loop, yielding every step as an ``AgentEvent`` so a UI can show it live
  3. remember the conversation, so follow-up questions have context

Findings and user context are saved by the model itself, through memory tools.
"""

from typing import Any, Iterator, Optional

from agents.base import Agent, AgentEvent, InvestigationResult
from agents.limits import Limits, get_limits
from agents.loop import DEFAULT_MAX_OUTPUT_TOKENS, resolve_llm, run_tool_loop, tool_log_entry
from agents.toolbox import Toolbox
from agents.usage import UsageMeter
from memory import ConversationMessage, MemoryStore
from prompts.single_agent import build_memory_context, get_single_agent_system_prompt

HISTORY_LIMIT = 20  # earlier chat messages of this session sent to the model


class SingleAgent(Agent):
    """
    Single transparent agent for repository investigation.

    Uses OpenAI gpt-4o-mini for tool-calling decisions with explicit
    reasoning and memory updates visible throughout the process.
    """

    def __init__(self, store: MemoryStore, client: Any = None, model: Optional[str] = None, max_steps: Optional[int] = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, limits: Optional[Limits] = None, identity: Optional[str] = None):
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
        """
        super().__init__("single_agent")
        self.store, self.max_output_tokens, self.identity = store, max_output_tokens, identity
        self.limits = limits or get_limits()
        self.max_steps = max_steps or self.limits.single_agent_steps
        self.usage = UsageMeter()  # tokens and estimated cost across everything this agent has run
        self._client, self._model = client, model

    def start_session(self, owner: str, repo: str) -> str:
        return self.store.create_session(MemoryStore.make_repo_id(owner, repo), "single_agent", identity=self.identity)

    def investigate(self, owner: str, repo: str, query: Optional[str] = None, session_id: Optional[str] = None) -> InvestigationResult:
        """Run to completion and return everything (use ``run`` to watch it live)."""
        session_id = session_id or self.start_session(owner, repo)
        events = list(self.run(owner, repo, query or "Analyze this repository.", session_id))
        last = events[-1]
        return InvestigationResult(
            session_id=session_id,
            answer=last.content if last.kind == "final" else "",
            events=events,
            steps=max((e.step for e in events), default=0),
            error=last.content if last.kind == "error" else None,
        )

    def run(self, owner: str, repo: str, query: str, session_id: str) -> Iterator[AgentEvent]:
        """The agent loop. Yields events as they happen; the last one is 'final' or 'error'."""
        try:
            client, model = resolve_llm(self._client, self._model)
        except Exception as exc:
            yield AgentEvent(kind="error", is_error=True, content=f"Could not start the OpenAI client: {exc}")
            return

        repo_id = MemoryStore.make_repo_id(owner, repo)
        toolbox = Toolbox(owner, repo, self.store, session_id, agent_name=self.agent_id, limits=self.limits)
        system = get_single_agent_system_prompt(self.max_steps) + "\n\n" + build_memory_context(
            self.store.get_resume_context(repo_id, identity=self.identity))
        history = self.store.get_conversation_history(session_id, limit=HISTORY_LIMIT)
        messages: list[dict] = [{"role": "system", "content": system}]
        messages += [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": query})
        self.store.save_conversation_message(ConversationMessage(session_id=session_id, role="user", content=query))

        tool_log: list[str] = []
        for event in run_tool_loop(client, model, messages, toolbox, max_steps=self.max_steps,
                                 max_output_tokens=self.max_output_tokens, usage=self.usage):
            if event.kind == "tool_result":
                tool_log.append(tool_log_entry(event))
            elif event.kind == "final":
                self.store.save_conversation_message(
                    ConversationMessage(session_id=session_id, role="assistant", content=event.content, tool_calls=tool_log or None)
                )
            yield event
