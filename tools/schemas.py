"""
Pydantic schemas for GitHub tool inputs and outputs.

These models ensure type safety and documentation for all tool operations.
Every tool returns either its ``*Output`` model or a ``ToolError`` so callers
(and the LLM) can always reason about failures instead of catching exceptions.
"""

import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

_OWNER_RE = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"
_REPO_RE = r"^[A-Za-z0-9._-]{1,100}$"

DEFAULT_MAX_CHARS = 20_000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

ErrorKind = Literal[
    "invalid_input",
    "not_found",
    "rate_limited",
    "auth",
    "forbidden",
    "empty_repository",
    "too_large",
    "binary_file",
    "network",
    "api_error",
    "unexpected",
]


class ToolError(BaseModel):
    """Structured failure returned by a tool instead of raising."""
    error: Literal[True] = True
    kind: ErrorKind
    message: str = Field(..., description="Human/LLM-readable explanation of what went wrong")
    status_code: Optional[int] = None
    retry_after_seconds: Optional[int] = Field(
        None, description="For rate limits: seconds until the request can be retried"
    )


def is_error(result: object) -> bool:
    """True if a tool result is a ToolError."""
    return isinstance(result, ToolError)


# ---------------------------------------------------------------------------
# Repository references
# ---------------------------------------------------------------------------

class RepoRef(BaseModel):
    """Base for every tool input: identifies one public repository."""
    owner: str = Field(..., pattern=_OWNER_RE, description="Repository owner username")
    repo: str = Field(..., pattern=_REPO_RE, description="Repository name")

    @field_validator("repo")
    @classmethod
    def _repo_not_dot_segment(cls, value: str) -> str:
        if value in (".", ".."):
            raise ValueError("repo must not be '.' or '..'")
        return value


def parse_repo_ref(ref: str) -> RepoRef:
    """
    Parse ``owner/repo`` or a GitHub URL into a RepoRef.

    Accepts ``owner/repo``, ``github.com/owner/repo``, and full URLs including
    ``.git`` suffixes and trailing paths such as ``/tree/main/src``.

    Raises:
        ValueError: if the text does not identify a repository.
    """
    text = ref.strip()
    text = re.sub(r"^(?:https?://)?(?:www\.)?github\.com/", "", text, flags=re.IGNORECASE)
    parts = [p for p in text.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse a repository from {ref!r}; expected 'owner/repo' or a GitHub URL")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    try:
        return RepoRef(owner=owner, repo=repo)
    except ValueError as exc:
        raise ValueError(f"{ref!r} is not a valid GitHub repository reference") from exc


# ---------------------------------------------------------------------------
# get_repository
# ---------------------------------------------------------------------------

class RepositoryInput(RepoRef):
    """Input for get_repository tool."""


class RepositoryOutput(BaseModel):
    """Output from get_repository tool."""
    name: str
    owner: str
    description: Optional[str] = None
    stars: int
    forks: int = 0
    open_issues_and_prs: int = Field(0, description="Open issues PLUS open pull requests, combined (GitHub counts them together)")
    language: Optional[str] = None
    topics: List[str] = []
    license: Optional[str] = None
    default_branch: str
    created_at: str
    updated_at: str
    pushed_at: Optional[str] = None
    is_fork: bool
    is_archived: bool = False
    size_kb: int = 0
    html_url: Optional[str] = None


# ---------------------------------------------------------------------------
# get_repository_tree
# ---------------------------------------------------------------------------

class RepositoryTreeInput(RepoRef):
    """Input for get_repository_tree tool."""
    path: Optional[str] = Field(None, description="Directory within repository (default: root)")
    recursive: bool = Field(False, description="Include subdirectories recursively")
    max_depth: Optional[int] = Field(
        None, ge=1, le=20,
        description="Maximum depth below `path` when recursive (1 = direct children only)",
    )
    max_entries: int = Field(500, ge=1, le=2000, description="Cap on returned nodes to protect LLM context")


class FileTreeNode(BaseModel):
    """A single node in the repository tree."""
    path: str
    type: Literal["file", "dir"]
    size: Optional[int] = None


class RepositoryTreeOutput(BaseModel):
    """Output from get_repository_tree tool. (Flags come first: a too-long result is cut from the end.)"""
    truncated: bool = Field(False, description="True if the tree was cut short (API limit or max_entries): more files exist than are listed")
    tree: List[FileTreeNode]


# ---------------------------------------------------------------------------
# get_file_content
# ---------------------------------------------------------------------------

class FileContentInput(RepoRef):
    """Input for get_file_content tool."""
    path: str = Field(..., min_length=1, description="Path to file in repository")
    max_chars: int = Field(DEFAULT_MAX_CHARS, ge=100, le=200_000, description="Truncate content beyond this many characters")


class FileContentOutput(BaseModel):
    """Output from get_file_content tool. (Flags come first: a too-long result is cut from the end.)"""
    path: str
    size: int = Field(..., description="Full file size in bytes (before truncation)")
    truncated: bool = Field(False, description="True if `content` is only the beginning of the file")
    content: str


# ---------------------------------------------------------------------------
# get_issues
# ---------------------------------------------------------------------------

class IssueInput(RepoRef):
    """Input for get_issues tool."""
    state: Literal["open", "closed", "all"] = Field("open", description="Filter by state")
    labels: Optional[List[str]] = Field(None, description="Only issues carrying all of these labels")
    oldest_first: bool = Field(False, description="Sort oldest first instead of newest first (use limit 1 to find the oldest open issue)")
    limit: int = Field(30, ge=1, le=100, description="Maximum number of issues to return")


class Issue(BaseModel):
    """Represents a GitHub issue (pull requests are excluded)."""
    number: int
    title: str
    state: str
    labels: List[str] = []
    author: Optional[str] = None
    comments: int = 0
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None


class IssuesOutput(BaseModel):
    """Output from get_issues tool."""
    note: Optional[str] = Field(None, description="Set when the list is capped: the true total is larger than `returned`")
    returned: int = Field(..., description="Number of issues in this response (NOT the repository's total)")
    has_more: bool = Field(False, description="True if more matching issues exist beyond `limit`")
    issues: List[Issue]


# ---------------------------------------------------------------------------
# get_pull_requests
# ---------------------------------------------------------------------------

class PullRequestInput(RepoRef):
    """Input for get_pull_requests tool."""
    state: Literal["open", "closed", "merged", "all"] = Field(
        "open", description="Filter by state; 'closed' means closed without merging"
    )
    oldest_first: bool = Field(False, description="Sort oldest first instead of newest first (use limit 1 to find the oldest open PR)")
    limit: int = Field(30, ge=1, le=100, description="Maximum number of PRs to return")


class PullRequest(BaseModel):
    """Represents a GitHub pull request."""
    number: int
    title: str
    state: str = Field(..., description="'open', 'closed' or 'merged'")
    author: Optional[str] = None
    draft: bool = False
    created_at: str
    updated_at: str
    merged_at: Optional[str] = None


class PullRequestsOutput(BaseModel):
    """Output from get_pull_requests tool."""
    note: Optional[str] = Field(None, description="Set when the list is capped: the true total is larger than `returned`")
    returned: int = Field(..., description="Number of pull requests in this response (NOT the repository's total)")
    has_more: bool = False
    pull_requests: List[PullRequest]


# ---------------------------------------------------------------------------
# get_commits
# ---------------------------------------------------------------------------

class CommitInput(RepoRef):
    """Input for get_commits tool."""
    limit: int = Field(30, ge=1, le=100, description="Maximum number of commits to return")
    since: Optional[str] = Field(None, description="ISO 8601 date string; only commits after this")


class Commit(BaseModel):
    """Represents a commit in the repository."""
    sha: str
    message: str = Field(..., description="Commit message (truncated)")
    author: str
    timestamp: str


class CommitsOutput(BaseModel):
    """Output from get_commits tool."""
    note: Optional[str] = Field(None, description="States whether the list is complete or capped; read it before quoting any count")
    returned: int = Field(0, description="Number of items in this response (NOT necessarily the repository's total)")
    has_more: bool = Field(False, description="True if more items exist beyond `limit`")
    commits: List[Commit]


# ---------------------------------------------------------------------------
# get_releases
# ---------------------------------------------------------------------------

class ReleaseInput(RepoRef):
    """Input for get_releases tool."""
    limit: int = Field(30, ge=1, le=100, description="Maximum number of releases to return")


class Release(BaseModel):
    """Represents a GitHub release."""
    tag_name: str
    name: Optional[str] = None
    published_at: Optional[str] = None
    prerelease: bool = False
    asset_count: int = 0
    body: Optional[str] = Field(None, description="Start of the release notes only")


class ReleasesOutput(BaseModel):
    """Output from get_releases tool."""
    note: Optional[str] = Field(None, description="States whether the list is complete or capped; read it before quoting any count")
    returned: int = Field(0, description="Number of items in this response (NOT necessarily the repository's total)")
    has_more: bool = Field(False, description="True if more items exist beyond `limit`")
    releases: List[Release]


# ---------------------------------------------------------------------------
# get_contributors
# ---------------------------------------------------------------------------

class ContributorInput(RepoRef):
    """Input for get_contributors tool."""
    limit: int = Field(30, ge=1, le=100, description="Maximum number of contributors to return")


class Contributor(BaseModel):
    """Represents a repository contributor."""
    login: str
    contributions: int


class ContributorsOutput(BaseModel):
    """Output from get_contributors tool."""
    note: Optional[str] = Field(None, description="States whether the list is complete or capped; read it before quoting any count")
    returned: int = Field(0, description="Number of items in this response (NOT necessarily the repository's total)")
    has_more: bool = Field(False, description="True if more items exist beyond `limit`")
    total_contributions: int = Field(0, description="Sum of contributions across the contributors returned")
    top_contributor_share_pct: float = Field(
        0.0, description="Top contributor's percentage of `total_contributions` (only among the contributors returned)"
    )
    contributors: List[Contributor]


# ---------------------------------------------------------------------------
# get_dependency_files
# ---------------------------------------------------------------------------

class DependencyFilesInput(RepoRef):
    """Input for get_dependency_files tool."""
    max_chars_per_file: int = Field(8_000, ge=100, le=50_000, description="Truncate each file beyond this")


class DependencyFile(BaseModel):
    """Represents a dependency file found in the repository."""
    path: str
    type: str  # "npm", "python", "go", "rust", "java", etc.
    truncated: bool = False
    content: str


class DependencyFilesOutput(BaseModel):
    """Output from get_dependency_files tool."""
    detected_languages: List[str]
    skipped: List[str] = Field([], description="Manifest paths found but not fetched (cap or fetch error)")
    files: List[DependencyFile]


# ---------------------------------------------------------------------------
# search_code (optional tool)
# ---------------------------------------------------------------------------

class SearchCodeInput(RepoRef):
    """Input for search_code tool. Requires GITHUB_TOKEN (GitHub code search needs auth)."""
    query: str = Field(..., min_length=1, max_length=256, description="Search terms")
    limit: int = Field(10, ge=1, le=30, description="Maximum number of matching files")


class CodeMatch(BaseModel):
    """A file matching a code search, with text fragments."""
    path: str
    fragments: List[str] = []


class SearchCodeOutput(BaseModel):
    """Output from search_code tool."""
    matches: List[CodeMatch]
    total_count: int = Field(..., description="Total matches GitHub reports (may exceed those returned)")


# Every tool returns its output or a ToolError.
ToolResult = Union[
    RepositoryOutput,
    RepositoryTreeOutput,
    FileContentOutput,
    IssuesOutput,
    PullRequestsOutput,
    CommitsOutput,
    ReleasesOutput,
    ContributorsOutput,
    DependencyFilesOutput,
    SearchCodeOutput,
    ToolError,
]
