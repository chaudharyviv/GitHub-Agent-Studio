"""
Base class and protocols for agents.

Defines the common interface and responsibility structure for all agent types
in the system (single agent, specialists, manager), plus the event stream
agents emit so the UI can show every decision as it happens.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """One visible step of an investigation."""
    kind: Literal["reasoning", "tool_call", "tool_result", "memory", "final", "error"]
    step: int = 0  # tool-calling round this event belongs to (0 = outside the loop)
    name: Optional[str] = None  # tool name
    arguments: Optional[dict] = None  # tool arguments as the model supplied them
    content: str = ""  # reasoning text, result summary, memory note, answer, or error message
    data: Any = None  # tool result payload (JSON-serializable)
    is_error: bool = False


def final_or_error(events: Sequence[AgentEvent]) -> Optional[AgentEvent]:
    """
    The event that decides a run's outcome: the last 'final' or 'error' in the stream.

    Usually ``events[-1]``, but not always: ``SingleAgent`` can yield a few more events (a silent
    save-finding nudge) after its real answer, so callers should use this instead of assuming the
    outcome is whatever came last.
    """
    return next((e for e in reversed(events) if e.kind in ("final", "error")), None)


class InvestigationResult(BaseModel):
    """Everything an investigation produced."""
    session_id: str
    answer: str
    events: List[AgentEvent] = Field(default_factory=list)
    steps: int = 0  # tool-calling rounds used
    error: Optional[str] = None


class Agent(ABC):
    """
    Abstract base class for all agents.

    All agents follow a similar structure:
    - Receive input (repository, context, user query)
    - Decide which tools to call and when to stop
    - Write findings to shared memory
    - Return structured results
    """

    def __init__(self, agent_id: str):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier for this agent (e.g., "single_agent", "architecture_specialist")
        """
        self.agent_id = agent_id

    @abstractmethod
    def investigate(self, owner: str, repo: str, query: Optional[str] = None) -> Any:
        """
        Conduct investigation into a repository.

        This is the main entry point for all agent investigation work.

        Args:
            owner: GitHub repository owner
            repo: GitHub repository name
            query: Optional user query or investigation directive

        Returns:
            Structured result from the investigation (format depends on agent type)
        """
        raise NotImplementedError("Subclasses must implement investigate()")
