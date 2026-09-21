"""
Pydantic schemas for memory storage.

These models define the structure of findings, repository profiles, user context,
and other persistent information stored in the SQLite database.

A repository is identified everywhere by its ``repo_id``: the lower-cased
``"owner/name"`` string (GitHub names are case-insensitive).
"""

import json
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Severity = Literal["info", "warning", "critical"]
Mode = Literal["single_agent", "multi_agent"]
Role = Literal["user", "assistant"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RepositoryProfile(BaseModel):
    """
    A stored profile of a repository from a previous investigation.

    Contains cached metadata and investigation summary for quick retrieval
    across sessions.
    """
    owner: str
    name: str
    description: Optional[str] = None
    stars: int
    primary_language: Optional[str] = None
    topics: List[str] = []
    license: Optional[str] = None
    default_branch: str
    last_analyzed: datetime = Field(default_factory=utcnow)
    profile_json: str = "{}"  # Serialized full metadata


class Finding(BaseModel):
    """
    A single finding or observation made by an agent.

    Findings are structured evidence entries that can be queried, filtered,
    and referenced by other agents or in final reports.
    """
    id: Optional[int] = None  # assigned by the store
    repo_id: Optional[str] = None  # filled in by the store when reading
    session_id: str
    agent: str  # e.g., "single_agent", "architecture_specialist", "security_specialist"
    severity: Severity
    category: str  # e.g., "architecture", "dependency_risk", "code_quality"
    finding: str  # Main text of the finding
    evidence_json: str = "{}"  # Structured evidence as JSON
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("evidence_json")
    @classmethod
    def _must_be_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except ValueError as exc:
            raise ValueError(f"evidence_json must be valid JSON: {exc}") from exc
        return value


class UserContext(BaseModel):
    """
    User-specified learning context or preference for a repository.

    Allows users to say "I'm learning this" and have the agent remember
    the focus area across sessions. Saving the same ``key`` again replaces
    the earlier value.
    """
    repository_id: str  # repo_id
    key: str  # e.g., "learning_focus", "preferred_language"
    value: str
    created_at: datetime = Field(default_factory=utcnow)


class ConversationMessage(BaseModel):
    """
    A single message in the conversation history.

    Stored to provide context and allow follow-up investigations
    across sessions.
    """
    session_id: str
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=utcnow)
    tool_calls: Optional[List[str]] = None  # JSON of tool calls made


class InvestigationSession(BaseModel):
    """
    A single investigation session for a repository.

    Groups findings and messages under a session for organization
    and session-level queries.
    """
    session_id: str
    repository_id: str  # repo_id
    mode: Mode
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
    metadata: Optional[str] = None  # Session-specific metadata as JSON


class ResumeContext(BaseModel):
    """Everything known about a repository, for "continue where we stopped"."""
    repo_id: str
    profile: Optional[RepositoryProfile] = None
    last_session: Optional[InvestigationSession] = None
    findings: List[Finding] = []  # newest first, across all sessions
    user_context: List[UserContext] = []
    recent_messages: List[ConversationMessage] = []  # chronological, across sessions
