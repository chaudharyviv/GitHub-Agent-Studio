"""
GitHub tools for accessing repository information.

Each tool takes one Pydantic input model and returns either its Pydantic output
model or a ``ToolError``. Tools never raise: API failures, rate limits, and even
unexpected bugs come back as structured errors the agent can reason about.

HTTP, auth, caching and rate-limit handling live in ``tools.client``.
"""

import base64
import functools
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import quote

from tools.client import GitHubAPIError, GitHubClient, get_client
from tools.schemas import (
    CodeMatch,
    Commit,
    CommitInput,
    CommitsOutput,
    Contributor,
    ContributorInput,
    ContributorsOutput,
    DependencyFile,
    DependencyFilesInput,
    DependencyFilesOutput,
    FileContentInput,
    FileContentOutput,
    FileTreeNode,
    Issue,
    IssueInput,
    IssuesOutput,
    PullRequest,
    PullRequestInput,
    PullRequestsOutput,
    Release,
    ReleaseInput,
    ReleasesOutput,
    RepositoryInput,
    RepositoryOutput,
    RepositoryTreeInput,
    RepositoryTreeOutput,
    SearchCodeInput,
    SearchCodeOutput,
    ToolError,
)

F = TypeVar("F", bound=Callable[..., Any])

TREE_TTL = 600.0
CONTENT_TTL = 900.0
ACTIVITY_TTL = 120.0  # issues / PRs / commits change often

MAX_COMMIT_MESSAGE_CHARS = 300
MAX_RELEASE_BODY_CHARS = 300  # cadence and versioning are what matter; full notes are large and rarely needed


def _tool(fn: F) -> F:
    """Turn any failure inside a tool into a ToolError instead of an exception."""

    @functools.wraps(fn)
    def wrapper(input_data):
        try:
            return fn(input_data)
        except GitHubAPIError as exc:
            return exc.to_tool_error()
        except Exception as exc:  # a tool must never crash the agent loop
            return ToolError(kind="unexpected", message=f"{fn.__name__} failed: {exc.__class__.__name__}: {exc}")

    return wrapper  # type: ignore[return-value]


def _repo_path(input_data) -> str:
    return f"/repos/{input_data.owner}/{input_data.repo}"


def _collect(
    client: GitHubClient,
    path: str,
    params: dict[str, Any],
    limit: int,
    *,
    keep: Optional[Callable[[dict], bool]] = None,
    max_pages: int = 5,
    ttl: float = ACTIVITY_TTL,
) -> tuple[list[dict], bool]:
    """
    Gather up to ``limit`` items from a paginated list endpoint.

    ``keep`` filters client-side (e.g. dropping PRs from the issues feed), which
    may need several pages. Returns (items, has_more).
    """
    per_page = 100 if keep else min(limit + 1, 100)  # +1 lets us detect "more exists" for free
    items: list[dict] = []
    for page in range(1, max_pages + 1):
        batch = client.get(path, {**params, "per_page": per_page, "page": page}, ttl=ttl) or []
        for item in batch:
            if keep and not keep(item):
                continue
            if len(items) == limit:
                return items, True
            items.append(item)
        if len(batch) < per_page:
            return items, False
        if not keep and len(items) >= limit:
            # limit == per_page (100 is GitHub's own page-size cap), so the "+1" trick above had no room on
            # this page; check for one more item instead of paying for a whole extra page to find out.
            extra = client.get(path, {**params, "per_page": 1, "page": limit + 1}, ttl=ttl) or []
            return items, bool(extra)
    return items, True


def _list_note(what: str, returned: int, has_more: bool) -> str:
    """Plain-language statement of whether a list is the whole thing, so a count of it is never mistaken for a total."""
    if has_more:
        return (f"Only {returned} {what} are shown. There are MORE than {returned}, so do not report {returned} as the total.")
    return f"All {returned} matching {what} are shown: this is the complete list, so {returned} is the exact total."


# ---------------------------------------------------------------------------
# get_repository
# ---------------------------------------------------------------------------

@_tool
def get_repository(input_data: RepositoryInput) -> RepositoryOutput | ToolError:
    """
    Fetch basic repository metadata.

    Returns repository name, description, star count, primary language,
    topics, license, default branch, and timestamps.
    """
    data = get_client().get(_repo_path(input_data))
    license_info = data.get("license") or {}
    return RepositoryOutput(
        name=data["name"],
        owner=data["owner"]["login"],
        description=data.get("description"),
        stars=data["stargazers_count"],
        forks=data.get("forks_count", 0),
        open_issues_and_prs=data.get("open_issues_count", 0),
        language=data.get("language"),
        topics=data.get("topics") or [],
        license=license_info.get("spdx_id") or license_info.get("name"),
        default_branch=data["default_branch"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        pushed_at=data.get("pushed_at"),
        is_fork=data["fork"],
        is_archived=data.get("archived", False),
        size_kb=data.get("size", 0),
        html_url=data.get("html_url"),
    )


# ---------------------------------------------------------------------------
# get_repository_tree
# ---------------------------------------------------------------------------

def _full_tree(client: GitHubClient, input_data) -> tuple[list[dict], bool]:
    """The whole recursive git tree of the default branch (cached), plus GitHub's truncated flag."""
    data = client.get(f"{_repo_path(input_data)}/git/trees/HEAD", {"recursive": "1"}, ttl=TREE_TTL)
    return data.get("tree", []), bool(data.get("truncated"))


@_tool
def get_repository_tree(input_data: RepositoryTreeInput) -> RepositoryTreeOutput | ToolError:
    """
    Fetch the file and directory structure of a repository.

    Lists the direct children of ``path`` (default: root). With ``recursive`` it
    descends up to ``max_depth`` levels. The full tree is fetched once and cached,
    then filtered locally, so exploring several directories costs one API call.
    """
    entries, api_truncated = _full_tree(get_client(), input_data)
    prefix = (input_data.path or "").strip("/")

    nodes: list[tuple[int, FileTreeNode]] = []  # (depth below prefix, node)
    prefix_is_file = False
    prefix_found = not prefix
    for entry in entries:
        path, kind = entry["path"], entry["type"]
        if kind not in ("blob", "tree"):
            continue  # submodule pointers
        if prefix:
            if path == prefix:
                prefix_is_file = kind == "blob"
                prefix_found = True
                continue
            if not path.startswith(prefix + "/"):
                continue
            relative = path[len(prefix) + 1:]
        else:
            relative = path
        depth = relative.count("/") + 1
        if not input_data.recursive and depth > 1:
            continue
        if input_data.recursive and input_data.max_depth and depth > input_data.max_depth:
            continue
        nodes.append((depth, FileTreeNode(
            path=path,
            type="file" if kind == "blob" else "dir",
            size=entry.get("size"),
        )))

    if prefix_is_file:
        raise GitHubAPIError("invalid_input", f"'{prefix}' is a file, not a directory. Use get_file_content instead.")
    if not prefix_found:
        raise GitHubAPIError("not_found", f"Directory '{prefix}' does not exist in this repository.", 404)

    truncated = api_truncated
    if len(nodes) > input_data.max_entries:
        nodes.sort(key=lambda item: item[0])  # stable: keep shallow entries when cutting
        nodes = nodes[: input_data.max_entries]
        truncated = True
    return RepositoryTreeOutput(
        tree=[node for _, node in sorted(nodes, key=lambda item: item[1].path)],
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# get_file_content
# ---------------------------------------------------------------------------

def _read_file(client: GitHubClient, owner: str, repo: str, path: str, max_chars: int) -> FileContentOutput:
    clean = path.strip("/")
    data = client.get(f"/repos/{owner}/{repo}/contents/{quote(clean, safe='/')}", ttl=CONTENT_TTL)

    if isinstance(data, list):
        raise GitHubAPIError("invalid_input", f"'{clean}' is a directory. Use get_repository_tree to list it.")
    if data.get("type") != "file":
        raise GitHubAPIError("invalid_input", f"'{clean}' is a {data.get('type')}, not a regular file.")

    size = data.get("size", 0)
    if data.get("encoding") != "base64" or (size and not data.get("content")):
        raise GitHubAPIError("too_large", f"'{clean}' is {size:,} bytes, over GitHub's 1 MB limit for this endpoint.")

    raw = base64.b64decode(data["content"])
    if b"\x00" in raw[:8192]:
        raise GitHubAPIError("binary_file", f"'{clean}' looks like a binary file ({size:,} bytes); not returning its content.")

    text = raw.decode("utf-8", errors="replace")
    return FileContentOutput(
        path=data.get("path", clean),
        content=text[:max_chars],
        size=size,
        truncated=len(text) > max_chars,
    )


@_tool
def get_file_content(input_data: FileContentInput) -> FileContentOutput | ToolError:
    """
    Fetch the raw content of a single file from the repository.

    Text is truncated to ``max_chars`` characters (``truncated`` is set when that
    happens). Binary files and files over 1 MB return an explanatory error.
    """
    return _read_file(get_client(), input_data.owner, input_data.repo, input_data.path, input_data.max_chars)


# ---------------------------------------------------------------------------
# get_dependency_files
# ---------------------------------------------------------------------------

_MANIFESTS = {
    "package.json": "npm",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "pipfile": "python",
    "setup.py": "python",
    "go.mod": "go",
    "cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "gemfile": "ruby",
    "composer.json": "php",
    "pubspec.yaml": "dart",
    "mix.exs": "elixir",
    "package.swift": "swift",
}
_ECOSYSTEM_LANGUAGE = {
    "npm": "JavaScript/TypeScript",
    "python": "Python",
    "go": "Go",
    "rust": "Rust",
    "java": "Java/Kotlin",
    "ruby": "Ruby",
    "php": "PHP",
    "dart": "Dart",
    "elixir": "Elixir",
    "swift": "Swift",
}
_IGNORED_DIRS = {"node_modules", "vendor", "third_party", ".venv", "venv", "testdata", "fixtures", "__fixtures__", "test", "tests", "example", "examples"}
_MAX_MANIFEST_DEPTH = 4  # path segments, so packages/foo/bar/package.json is the deepest
_MAX_MANIFESTS = 12


def _classify_manifest(path: str) -> Optional[str]:
    segments = path.split("/")
    if len(segments) > _MAX_MANIFEST_DEPTH or any(s in _IGNORED_DIRS for s in segments[:-1]):
        return None
    name = segments[-1].lower()
    if name in _MANIFESTS:
        return _MANIFESTS[name]
    if name.startswith("requirements") and name.endswith(".txt"):
        return "python"
    return None


@_tool
def get_dependency_files(input_data: DependencyFilesInput) -> DependencyFilesOutput | ToolError:
    """
    Detect and fetch common dependency files from the repository.

    Looks for package.json (npm), requirements.txt / pyproject.toml (Python),
    go.mod (Go), Cargo.toml (Rust), pom.xml / build.gradle (Java), and other
    manifests, shallowest first. Vendored and test directories are ignored.
    """
    client = get_client()
    entries, _ = _full_tree(client, input_data)

    found = []
    for entry in entries:
        if entry["type"] != "blob":
            continue
        kind = _classify_manifest(entry["path"])
        if kind:
            found.append((entry["path"], kind))
    found.sort(key=lambda item: (item[0].count("/"), item[0]))

    files: list[DependencyFile] = []
    skipped: list[str] = [path for path, _ in found[_MAX_MANIFESTS:]]
    for path, kind in found[:_MAX_MANIFESTS]:
        try:
            content = _read_file(client, input_data.owner, input_data.repo, path, input_data.max_chars_per_file)
        except GitHubAPIError as exc:
            if exc.kind == "rate_limited":
                raise
            skipped.append(path)
            continue
        files.append(DependencyFile(path=path, type=kind, content=content.content, truncated=content.truncated))

    languages: list[str] = []
    for file in files:
        language = _ECOSYSTEM_LANGUAGE.get(file.type, file.type)
        if language not in languages:
            languages.append(language)
    return DependencyFilesOutput(files=files, detected_languages=languages, skipped=skipped)


# ---------------------------------------------------------------------------
# get_issues / get_pull_requests
# ---------------------------------------------------------------------------

@_tool
def get_issues(input_data: IssueInput) -> IssuesOutput | ToolError:
    """
    Fetch issues from the repository with filtering.

    Filters by state and labels. GitHub's issues feed also contains pull
    requests; those are removed here (use get_pull_requests for them).
    """
    params: dict[str, Any] = {"state": input_data.state}
    if input_data.oldest_first:
        params.update(sort="created", direction="asc")
    if input_data.labels:
        params["labels"] = ",".join(input_data.labels)
    items, has_more = _collect(
        get_client(), f"{_repo_path(input_data)}/issues", params, input_data.limit,
        keep=lambda item: "pull_request" not in item,
    )
    issues = [
        Issue(
            number=item["number"],
            title=item["title"],
            state=item["state"],
            labels=[label["name"] for label in item.get("labels", []) if isinstance(label, dict)],
            author=(item.get("user") or {}).get("login"),
            comments=item.get("comments", 0),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            closed_at=item.get("closed_at"),
        )
        for item in items
    ]
    return IssuesOutput(issues=issues, returned=len(issues), has_more=has_more, note=_list_note("issues", len(issues), has_more))


@_tool
def get_pull_requests(input_data: PullRequestInput) -> PullRequestsOutput | ToolError:
    """
    Fetch pull requests from the repository with filtering.

    States are mutually exclusive: 'open', 'merged', 'closed' (closed without
    merging), or 'all'.
    """
    wanted = input_data.state
    keep = {
        "merged": lambda pr: bool(pr.get("merged_at")),
        "closed": lambda pr: not pr.get("merged_at"),
    }.get(wanted)
    items, has_more = _collect(
        get_client(), f"{_repo_path(input_data)}/pulls",
        {"state": "closed" if wanted in ("merged", "closed") else wanted, **({"sort": "created", "direction": "asc"} if input_data.oldest_first else {})},
        input_data.limit, keep=keep,
    )
    prs = [
        PullRequest(
            number=item["number"],
            title=item["title"],
            state="merged" if item.get("merged_at") else item["state"],
            author=(item.get("user") or {}).get("login"),
            draft=item.get("draft", False),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            merged_at=item.get("merged_at"),
        )
        for item in items
    ]
    return PullRequestsOutput(pull_requests=prs, returned=len(prs), has_more=has_more, note=_list_note("pull requests", len(prs), has_more))


# ---------------------------------------------------------------------------
# get_commits / get_releases / get_contributors
# ---------------------------------------------------------------------------

@_tool
def get_commits(input_data: CommitInput) -> CommitsOutput | ToolError:
    """
    Fetch recent commit history from the repository.

    Newest first, optionally only commits after ``since``. An empty repository
    yields an empty list rather than an error.
    """
    params = {"since": input_data.since} if input_data.since else {}
    try:
        items, has_more = _collect(get_client(), f"{_repo_path(input_data)}/commits", params, input_data.limit)
    except GitHubAPIError as exc:
        if exc.kind == "empty_repository":
            return CommitsOutput(commits=[], note=_list_note("commits", 0, False))
        raise
    commits = [
        Commit(
            sha=item["sha"],
            message=item["commit"]["message"][:MAX_COMMIT_MESSAGE_CHARS],
            author=item["commit"]["author"].get("name") or (item.get("author") or {}).get("login") or "unknown",
            timestamp=item["commit"]["author"]["date"],
        )
        for item in items
    ]
    return CommitsOutput(commits=commits, returned=len(commits), has_more=has_more, note=_list_note("commits", len(commits), has_more))


@_tool
def get_releases(input_data: ReleaseInput) -> ReleasesOutput | ToolError:
    """
    Fetch published releases from the repository.

    Returns tag names, release names, publish dates, release notes (truncated)
    and asset metadata. Used to assess versioning strategy and release frequency.
    """
    items, has_more = _collect(get_client(), f"{_repo_path(input_data)}/releases", {}, input_data.limit)
    releases = [
        Release(
            tag_name=item["tag_name"],
            name=item.get("name"),
            published_at=item.get("published_at"),
            prerelease=item.get("prerelease", False),
            asset_count=len(item.get("assets", [])),
            body=(item.get("body") or "")[:MAX_RELEASE_BODY_CHARS] or None,
        )
        for item in items
    ]
    return ReleasesOutput(releases=releases, returned=len(releases), has_more=has_more, note=_list_note("releases", len(releases), has_more))


@_tool
def get_contributors(input_data: ContributorInput) -> ContributorsOutput | ToolError:
    """
    Fetch top contributors to the repository.

    Returns contributor login and contribution counts, most active first.
    Used to understand team size and distribution of effort.
    """
    items, has_more = _collect(get_client(), f"{_repo_path(input_data)}/contributors", {}, input_data.limit, ttl=TREE_TTL)
    contributors = [Contributor(login=item.get("login", "unknown"), contributions=item["contributions"]) for item in items]
    total = sum(c.contributions for c in contributors)
    top_share = round(100 * max((c.contributions for c in contributors), default=0) / total, 1) if total else 0.0
    return ContributorsOutput(
        contributors=contributors, returned=len(contributors), has_more=has_more, total_contributions=total,
        top_contributor_share_pct=top_share, note=_list_note("contributors", len(contributors), has_more),
    )


# ---------------------------------------------------------------------------
# search_code (optional)
# ---------------------------------------------------------------------------

@_tool
def search_code(input_data: SearchCodeInput) -> SearchCodeOutput | ToolError:
    """
    Search code within the repository using GitHub code search.

    GitHub only allows code search for authenticated requests, so this returns an
    'auth' error when GITHUB_TOKEN is not set. Search has its own, stricter rate
    limit (about 10 requests/minute).
    """
    client = get_client()
    if not client.token:
        raise GitHubAPIError("auth", "search_code requires GITHUB_TOKEN; GitHub does not allow unauthenticated code search.")
    data = client.get(
        "/search/code",
        {"q": f"{input_data.query} repo:{input_data.owner}/{input_data.repo}", "per_page": input_data.limit},
        accept="application/vnd.github.text-match+json",
    )
    return SearchCodeOutput(
        matches=[
            CodeMatch(path=item["path"], fragments=[m["fragment"] for m in item.get("text_matches", []) if m.get("fragment")])
            for item in data.get("items", [])
        ],
        total_count=data.get("total_count", 0),
    )
