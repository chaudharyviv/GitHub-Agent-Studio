"""
SQLite-based memory storage for GitHub Agent Studio.

This module provides a clean interface for persisting and querying
repository profiles, findings, user context, and conversation history.

Everything hangs off a repository: ``repositories`` is the root table and every
other table cascades from it, so clearing a repository's memory is one DELETE.
Repositories are keyed by ``repo_id`` (lower-cased ``"owner/name"``).
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

from memory.schemas import (
    ConversationMessage,
    Finding,
    InvestigationSession,
    RepositoryProfile,
    ResumeContext,
    UserContext,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
    repo_id       TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    name          TEXT NOT NULL,
    last_analyzed TEXT,
    profile_json  TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    repo_id      TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    mode         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    completed_at TEXT,
    metadata     TEXT,
    identity     TEXT  -- optional user-chosen name; NULL means shared/anonymous (see MemoryStore.list_sessions)
);
CREATE TABLE IF NOT EXISTS findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    agent         TEXT NOT NULL,
    severity      TEXT NOT NULL,
    category      TEXT NOT NULL,
    finding       TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence    REAL NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    tool_calls_json TEXT
);
CREATE TABLE IF NOT EXISTS user_context (
    repo_id    TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (repo_id, key)
);
CREATE INDEX IF NOT EXISTS idx_sessions_repo     ON sessions(repo_id);
CREATE INDEX IF NOT EXISTS idx_findings_session  ON findings(session_id);
CREATE INDEX IF NOT EXISTS idx_findings_agent    ON findings(agent);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_messages_session  ON conversation_messages(session_id);
"""

_MODES = ("single_agent", "multi_agent")


def _to_text(dt: datetime) -> str:
    """Datetimes are stored as UTC ISO-8601 text (naive values are assumed to be UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_text(text: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(text) if text else None


class MemoryStore:
    """
    SQLite-backed persistent memory for agents and investigations.

    Handles all database initialization, queries, and writes for findings,
    repository profiles, user context, and session management. One connection
    is shared behind a lock, so a single store can be used from Streamlit's
    worker threads (and ``":memory:"`` databases work for tests).
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        """
        Initialize the memory store and create tables if needed.

        Args:
            db_path: Path to SQLite database file (``":memory:"`` for a throwaway store)
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_db()

    # -- plumbing ------------------------------------------------------------

    def _initialize_db(self) -> None:
        """Create all necessary tables if they don't exist."""
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(sessions)")}
            if "identity" not in columns:  # DB created before per-identity session scoping existed
                self._conn.execute("ALTER TABLE sessions ADD COLUMN identity TEXT")
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.db_path} was written by a newer version of the app "
                    f"(schema {version} > {SCHEMA_VERSION})"
                )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Locked transaction: commits on success, rolls back on error."""
        with self._lock:
            with self._conn:
                yield self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def make_repo_id(owner: str, name: str) -> str:
        """Canonical repository key: lower-cased ``owner/name``."""
        return f"{owner.strip()}/{name.strip()}".lower()

    @staticmethod
    def _norm(repo_id: str) -> str:
        repo_id = repo_id.strip().lower()
        owner, _, name = repo_id.partition("/")
        if not owner or not name or "/" in name:
            raise ValueError(f"repo_id must look like 'owner/name', got {repo_id!r}")
        return repo_id

    def _ensure_repo(self, conn: sqlite3.Connection, repo_id: str) -> str:
        repo_id = self._norm(repo_id)
        owner, _, name = repo_id.partition("/")
        conn.execute(
            "INSERT OR IGNORE INTO repositories (repo_id, owner, name) VALUES (?, ?, ?)",
            (repo_id, owner, name),
        )
        return repo_id

    # -- findings ------------------------------------------------------------

    def save_finding(self, finding: Finding) -> int:
        """
        Persist a single finding to the database.

        Args:
            finding: Finding object to save (its session must already exist)

        Returns:
            Finding ID (primary key)

        Raises:
            ValueError: if ``finding.session_id`` is not a known session
        """
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (finding.session_id,)).fetchone() is None:
                raise ValueError(f"Unknown session_id {finding.session_id!r}; call create_session first")
            cursor = conn.execute(
                "INSERT INTO findings (session_id, agent, severity, category, finding, evidence_json, confidence, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    finding.session_id, finding.agent, finding.severity, finding.category, finding.finding,
                    finding.evidence_json, finding.confidence, _to_text(finding.created_at),
                ),
            )
            return int(cursor.lastrowid)

    def get_findings(
        self,
        repo_id: str,
        agent: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Finding]:
        """
        Query findings for a repository across all its sessions, newest first.

        Args:
            repo_id: Repository identifier
            agent: Filter by agent name (optional)
            severity: Filter by severity level (optional)
            category: Filter by category (optional)
            session_id: Restrict to one session (optional)
            limit: Maximum number of findings to return (optional)

        Returns:
            List of Finding objects matching criteria
        """
        clauses, args = ["s.repo_id = ?"], [self._norm(repo_id)]
        for column, value in (("f.agent", agent), ("f.severity", severity), ("f.category", category), ("f.session_id", session_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                args.append(value)
        sql = (
            "SELECT f.*, s.repo_id FROM findings f JOIN sessions s ON s.session_id = f.session_id"
            f" WHERE {' AND '.join(clauses)} ORDER BY f.id DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._finding_from_row(row) for row in rows]

    @staticmethod
    def _finding_from_row(row: sqlite3.Row) -> Finding:
        return Finding(
            id=row["id"], repo_id=row["repo_id"], session_id=row["session_id"], agent=row["agent"],
            severity=row["severity"], category=row["category"], finding=row["finding"],
            evidence_json=row["evidence_json"], confidence=row["confidence"],
            created_at=_from_text(row["created_at"]),
        )

    # -- repository profile --------------------------------------------------

    def save_repository_profile(self, profile: RepositoryProfile) -> None:
        """
        Persist (insert or replace) a repository profile for future retrieval.

        Args:
            profile: RepositoryProfile object to save
        """
        repo_id = self.make_repo_id(profile.owner, profile.name)
        with self._tx() as conn:
            self._ensure_repo(conn, repo_id)
            conn.execute(
                "UPDATE repositories SET last_analyzed = ?, profile_json = ? WHERE repo_id = ?",
                (_to_text(profile.last_analyzed), profile.model_dump_json(), repo_id),
            )

    def get_repository_profile(self, owner: str, repo: str) -> Optional[RepositoryProfile]:
        """
        Retrieve a previously saved repository profile.

        Returns:
            RepositoryProfile if found, None otherwise
        """
        return self._profile(self.make_repo_id(owner, repo))

    def _profile(self, repo_id: str) -> Optional[RepositoryProfile]:
        with self._lock:
            row = self._conn.execute("SELECT profile_json FROM repositories WHERE repo_id = ?", (repo_id,)).fetchone()
        return RepositoryProfile.model_validate_json(row["profile_json"]) if row and row["profile_json"] else None

    def list_repositories(self) -> List[str]:
        """repo_ids that have any stored memory, most recently analyzed first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT repo_id FROM repositories ORDER BY last_analyzed IS NULL, last_analyzed DESC, repo_id"
            ).fetchall()
        return [row["repo_id"] for row in rows]

    # -- user context --------------------------------------------------------

    def save_user_context(self, context: UserContext) -> None:
        """
        Persist user learning context or preferences for a repository.

        Saving an existing ``key`` replaces its value.
        """
        with self._tx() as conn:
            repo_id = self._ensure_repo(conn, context.repository_id)
            conn.execute(
                "INSERT INTO user_context (repo_id, key, value, created_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(repo_id, key) DO UPDATE SET value = excluded.value, created_at = excluded.created_at",
                (repo_id, context.key, context.value, _to_text(context.created_at)),
            )

    def get_user_context(self, repo_id: str) -> List[UserContext]:
        """Retrieve all user context entries for a repository, oldest first."""
        repo_id = self._norm(repo_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM user_context WHERE repo_id = ? ORDER BY created_at, key", (repo_id,)
            ).fetchall()
        return [
            UserContext(repository_id=row["repo_id"], key=row["key"], value=row["value"], created_at=_from_text(row["created_at"]))
            for row in rows
        ]

    def delete_user_context(self, repo_id: str, key: str) -> bool:
        """Forget one context entry. Returns True if it existed."""
        with self._tx() as conn:
            return conn.execute("DELETE FROM user_context WHERE repo_id = ? AND key = ?", (self._norm(repo_id), key)).rowcount > 0

    # -- conversation --------------------------------------------------------

    def save_conversation_message(self, message: ConversationMessage) -> None:
        """
        Persist a conversation message to the database.

        Raises:
            ValueError: if ``message.session_id`` is not a known session
        """
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (message.session_id,)).fetchone() is None:
                raise ValueError(f"Unknown session_id {message.session_id!r}; call create_session first")
            conn.execute(
                "INSERT INTO conversation_messages (session_id, role, content, timestamp, tool_calls_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    message.session_id, message.role, message.content, _to_text(message.timestamp),
                    json.dumps(message.tool_calls) if message.tool_calls is not None else None,
                ),
            )

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[ConversationMessage]:
        """
        Retrieve messages from a conversation session in chronological order.

        Args:
            session_id: Session identifier
            limit: If given, only the most recent ``limit`` messages
        """
        return self._messages("WHERE session_id = ?", [session_id], limit)

    def _messages(self, where: str, args: list, limit: Optional[int]) -> List[ConversationMessage]:
        sql = f"SELECT m.* FROM conversation_messages m {where} ORDER BY m.id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args = [*args, limit]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            ConversationMessage(
                session_id=row["session_id"], role=row["role"], content=row["content"],
                timestamp=_from_text(row["timestamp"]),
                tool_calls=json.loads(row["tool_calls_json"]) if row["tool_calls_json"] is not None else None,
            )
            for row in reversed(rows)
        ]

    # -- sessions ------------------------------------------------------------

    def create_session(self, repository_id: str, mode: str, metadata: Optional[str] = None, identity: Optional[str] = None) -> str:
        """
        Create and persist a new investigation session.

        Args:
            repository_id: Repository being investigated (``owner/name``)
            mode: Investigation mode ("single_agent" or "multi_agent")
            metadata: Optional session-specific JSON
            identity: Optional user-chosen name that started this session (see ``list_sessions``)

        Returns:
            Session ID (UUID hex)
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        session_id = uuid.uuid4().hex
        with self._tx() as conn:
            repo_id = self._ensure_repo(conn, repository_id)
            conn.execute(
                "INSERT INTO sessions (session_id, repo_id, mode, created_at, metadata, identity) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, repo_id, mode, _to_text(datetime.now(timezone.utc)), metadata, identity),
            )
        return session_id

    def set_session_metadata(self, session_id: str, metadata: str) -> None:
        """Replace a session's metadata (JSON text), e.g. to store its final report."""
        with self._tx() as conn:
            updated = conn.execute("UPDATE sessions SET metadata = ? WHERE session_id = ?", (metadata, session_id)).rowcount
            if not updated:
                raise ValueError(f"Unknown session_id {session_id!r}")

    def complete_session(self, session_id: str) -> None:
        """Mark a session as complete with a timestamp."""
        with self._tx() as conn:
            updated = conn.execute(
                "UPDATE sessions SET completed_at = ? WHERE session_id = ?",
                (_to_text(datetime.now(timezone.utc)), session_id),
            ).rowcount
            if not updated:
                raise ValueError(f"Unknown session_id {session_id!r}")

    def get_session(self, session_id: str) -> Optional[InvestigationSession]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(
        self, repo_id: str, mode: Optional[str] = None, identity: Optional[str] = None, limit: Optional[int] = None
    ) -> List[InvestigationSession]:
        """
        Sessions for a repository, newest first.

        Args:
            identity: When given, only sessions started under this name, plus anonymous/legacy
                sessions (``identity IS NULL``) started before anyone set a name, or by someone who
                never set one. Findings and reports stay shared team knowledge; this only limits
                which *conversation* a caller may silently resume, so on a shared instance, one
                named person never drops into another named person's live chat.
        """
        sql, args = "SELECT * FROM sessions WHERE repo_id = ?", [self._norm(repo_id)]
        if mode is not None:
            sql += " AND mode = ?"
            args.append(mode)
        if identity is not None:
            sql += " AND (identity = ? OR identity IS NULL)"
            args.append(identity)
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._session_from_row(row) for row in rows]

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> InvestigationSession:
        return InvestigationSession(
            session_id=row["session_id"], repository_id=row["repo_id"], mode=row["mode"],
            created_at=_from_text(row["created_at"]), completed_at=_from_text(row["completed_at"]),
            metadata=row["metadata"], identity=row["identity"],
        )

    # -- resume / clear / export ---------------------------------------------

    def get_resume_context(
        self, repo_id: str, max_findings: int = 50, max_messages: int = 20, identity: Optional[str] = None
    ) -> ResumeContext:
        """
        Everything needed to "continue where we stopped" on a repository:
        profile, the latest session, recent findings and messages, and user context.

        Findings and profile are shared team knowledge, so they are never scoped. ``recent_messages``
        feeds straight into the model's context, so when ``identity`` is given it is restricted to
        this name's own conversation turns (plus anonymous/legacy messages) instead of whatever
        anyone last said to the agent.
        """
        repo_id = self._norm(repo_id)
        sessions = self.list_sessions(repo_id, identity=identity, limit=1)
        where = "JOIN sessions s ON s.session_id = m.session_id WHERE s.repo_id = ?"
        args: list = [repo_id]
        if identity is not None:
            where += " AND (s.identity = ? OR s.identity IS NULL)"
            args.append(identity)
        return ResumeContext(
            repo_id=repo_id,
            profile=self._profile(repo_id),
            last_session=sessions[0] if sessions else None,
            findings=self.get_findings(repo_id, limit=max_findings),
            user_context=self.get_user_context(repo_id),
            recent_messages=self._messages(where, args, max_messages),
        )

    def clear_repository_memory(self, repo_id: str) -> None:
        """
        Delete everything stored for a repository: profile, sessions, findings,
        conversation messages and user context.

        Useful for restarting investigation or user-initiated reset.
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM repositories WHERE repo_id = ?", (self._norm(repo_id),))  # cascades

    def export_repository_memory(self, repo_id: str) -> dict[str, Any]:
        """All memory for a repository as a JSON-serializable dict."""
        repo_id = self._norm(repo_id)
        sessions = self.list_sessions(repo_id)
        profile = self._profile(repo_id)
        return {
            "repo_id": repo_id,
            "exported_at": _to_text(datetime.now(timezone.utc)),
            "profile": profile.model_dump(mode="json") if profile else None,
            "user_context": [c.model_dump(mode="json") for c in self.get_user_context(repo_id)],
            "sessions": [s.model_dump(mode="json") for s in sessions],
            "findings": [f.model_dump(mode="json") for f in reversed(self.get_findings(repo_id))],
            "messages": [
                m.model_dump(mode="json")
                for s in reversed(sessions)
                for m in self.get_conversation_history(s.session_id)
            ],
        }
