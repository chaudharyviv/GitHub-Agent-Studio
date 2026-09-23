"""Offline tests for the GitHub tools layer, using httpx.MockTransport (no network)."""

import base64
import time

import httpx
import pytest

from tools import (
    get_commits,
    get_dependency_files,
    get_file_content,
    get_issues,
    get_pull_requests,
    get_repository,
    get_repository_tree,
    is_error,
    parse_repo_ref,
    search_code,
)
from tools.client import GitHubClient, set_client
from tools.schemas import (
    CommitInput,
    DependencyFilesInput,
    FileContentInput,
    IssueInput,
    PullRequestInput,
    RepositoryInput,
    RepositoryTreeInput,
    SearchCodeInput,
)

REPO = {"owner": "octo", "repo": "demo"}


def b64(text: str | bytes) -> str:
    raw = text.encode() if isinstance(text, str) else text
    return base64.encodebytes(raw).decode()  # includes newlines, like GitHub


TREE = [
    {"path": "README.md", "type": "blob", "size": 10, "sha": "a"},
    {"path": "package.json", "type": "blob", "size": 20, "sha": "b"},
    {"path": "src", "type": "tree", "sha": "c"},
    {"path": "src/main.py", "type": "blob", "size": 5, "sha": "d"},
    {"path": "src/pkg", "type": "tree", "sha": "e"},
    {"path": "src/pkg/util.py", "type": "blob", "size": 5, "sha": "f"},
    {"path": "node_modules/x/package.json", "type": "blob", "size": 5, "sha": "g"},
    {"path": "libs/core/pyproject.toml", "type": "blob", "size": 5, "sha": "h"},
    {"path": "sub", "type": "commit", "sha": "i"},
]

FILES = {
    "README.md": "# Demo\n" + "x" * 500,
    "package.json": '{"name": "demo"}',
    "libs/core/pyproject.toml": "[project]\nname='core'",
}


class FakeGitHub:
    """Routes requests to canned responses and records what was asked."""

    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.overrides: dict[str, httpx.Response] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path in self.overrides:
            return self.overrides[path]
        if path == "/repos/octo/demo":
            return httpx.Response(200, json={
                "name": "demo", "owner": {"login": "octo"}, "description": "A demo",
                "stargazers_count": 42, "language": "Python", "topics": ["a"],
                "license": {"spdx_id": "MIT"}, "default_branch": "main",
                "created_at": "2020-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
                "fork": False,
            })
        if path == "/repos/octo/demo/git/trees/HEAD":
            return httpx.Response(200, json={"tree": TREE, "truncated": False})
        if path.startswith("/repos/octo/demo/contents/"):
            name = path.removeprefix("/repos/octo/demo/contents/")
            if name == "src":
                return httpx.Response(200, json=[{"name": "main.py"}])
            if name == "logo.png":
                return httpx.Response(200, json={
                    "type": "file", "path": name, "size": 4, "encoding": "base64",
                    "content": b64(b"\x89PNG\x00\x01"), "sha": "z",
                })
            if name == "huge.bin":
                return httpx.Response(200, json={"type": "file", "path": name, "size": 5_000_000, "encoding": "none", "content": ""})
            if name in FILES:
                body = FILES[name]
                return httpx.Response(200, json={
                    "type": "file", "path": name, "size": len(body), "encoding": "base64",
                    "content": b64(body), "sha": "s",
                }, headers={"etag": '"v1"'})
            return httpx.Response(404, json={"message": "Not Found"})
        if path == "/repos/octo/demo/issues":
            page = int(request.url.params["page"])
            if page > 1:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[
                {"number": 1, "title": "bug", "state": "open", "labels": [{"name": "bug"}], "user": {"login": "u"},
                 "comments": 2, "created_at": "t", "updated_at": "t"},
                {"number": 2, "title": "a PR", "state": "open", "pull_request": {}, "created_at": "t", "updated_at": "t"},
                {"number": 3, "title": "feat", "state": "open", "labels": [], "user": None,
                 "created_at": "t", "updated_at": "t"},
            ])
        if path == "/repos/octo/demo/pulls":
            if int(request.url.params["page"]) > 1:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[
                {"number": 10, "title": "merged one", "state": "closed", "merged_at": "t", "user": {"login": "u"},
                 "created_at": "t", "updated_at": "t"},
                {"number": 11, "title": "abandoned", "state": "closed", "merged_at": None, "user": {"login": "u"},
                 "created_at": "t", "updated_at": "t"},
            ])
        if path == "/repos/octo/demo/commits":
            n = int(request.url.params["per_page"])
            return httpx.Response(200, json=[
                {"sha": f"sha{i}", "commit": {"message": "m" * 400, "author": {"name": "Ann", "date": "t"}}, "author": None}
                for i in range(n)
            ])
        if path == "/search/code":
            return httpx.Response(200, json={"total_count": 1, "items": [
                {"path": "src/main.py", "text_matches": [{"fragment": "def main()"}]}
            ]})
        return httpx.Response(404, json={"message": "Not Found"})


@pytest.fixture
def gh():
    fake = FakeGitHub()
    client = GitHubClient(token=None, transport=httpx.MockTransport(fake), sleep=lambda _: None)
    set_client(client)
    yield fake
    set_client(None)


# -- parsing -----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "octo/demo",
    "https://github.com/octo/demo",
    "https://github.com/octo/demo.git",
    "github.com/octo/demo/tree/main/src",
    "  http://www.github.com/octo/demo/ ",
])
def test_parse_repo_ref(text):
    ref = parse_repo_ref(text)
    assert (ref.owner, ref.repo) == ("octo", "demo")


@pytest.mark.parametrize("text", ["", "justone", "octo/../etc", "octo/de mo"])
def test_parse_repo_ref_rejects(text):
    with pytest.raises(ValueError):
        parse_repo_ref(text)


# -- repository / tree ---------------------------------------------------------

def test_get_repository(gh):
    result = get_repository(RepositoryInput(**REPO))
    assert result.stars == 42 and result.license == "MIT" and result.default_branch == "main"


def test_tree_root_is_shallow_and_skips_submodules(gh):
    result = get_repository_tree(RepositoryTreeInput(**REPO))
    paths = {n.path: n.type for n in result.tree}
    assert paths["src"] == "dir" and paths["README.md"] == "file"
    assert "src/main.py" not in paths and "sub" not in paths


def test_tree_subpath_recursive_with_depth(gh):
    result = get_repository_tree(RepositoryTreeInput(**REPO, path="src", recursive=True, max_depth=1))
    assert {n.path for n in result.tree} == {"src/main.py", "src/pkg"}
    deep = get_repository_tree(RepositoryTreeInput(**REPO, path="src", recursive=True))
    assert "src/pkg/util.py" in {n.path for n in deep.tree}


def test_tree_errors(gh):
    assert get_repository_tree(RepositoryTreeInput(**REPO, path="nope")).kind == "not_found"
    assert get_repository_tree(RepositoryTreeInput(**REPO, path="README.md")).kind == "invalid_input"


def test_tree_max_entries_keeps_shallow(gh):
    result = get_repository_tree(RepositoryTreeInput(**REPO, recursive=True, max_entries=3))
    assert result.truncated and len(result.tree) == 3
    assert all(n.path.count("/") == 0 for n in result.tree)


def test_tree_cached_across_calls(gh):
    get_repository_tree(RepositoryTreeInput(**REPO))
    get_repository_tree(RepositoryTreeInput(**REPO, path="src"))
    assert sum(c.url.path.endswith("/git/trees/HEAD") for c in gh.calls) == 1


# -- files ---------------------------------------------------------------------

def test_file_content_truncates(gh):
    result = get_file_content(FileContentInput(**REPO, path="README.md", max_chars=100))
    assert result.truncated and len(result.content) == 100 and result.size == len(FILES["README.md"])
    full = get_file_content(FileContentInput(**REPO, path="package.json"))
    assert not full.truncated and full.content == FILES["package.json"]


def test_file_content_error_cases(gh):
    assert get_file_content(FileContentInput(**REPO, path="src")).kind == "invalid_input"
    assert get_file_content(FileContentInput(**REPO, path="logo.png")).kind == "binary_file"
    assert get_file_content(FileContentInput(**REPO, path="huge.bin")).kind == "too_large"
    assert get_file_content(FileContentInput(**REPO, path="missing.txt")).kind == "not_found"


def test_dependency_files(gh):
    result = get_dependency_files(DependencyFilesInput(**REPO))
    assert [f.path for f in result.files] == ["package.json", "libs/core/pyproject.toml"]
    assert result.detected_languages == ["JavaScript/TypeScript", "Python"]
    assert result.skipped == []  # node_modules manifest ignored, not "skipped"


# -- issues / PRs / commits ------------------------------------------------------

def test_issues_exclude_pull_requests(gh):
    result = get_issues(IssueInput(**REPO))
    assert [i.number for i in result.issues] == [1, 3]
    assert result.issues[0].labels == ["bug"] and result.issues[1].author is None
    assert not result.has_more


def test_issues_limit_sets_has_more(gh):
    result = get_issues(IssueInput(**REPO, limit=1))
    assert [i.number for i in result.issues] == [1] and result.has_more


def test_pull_request_state_filters(gh):
    assert [p.number for p in get_pull_requests(PullRequestInput(**REPO, state="merged")).pull_requests] == [10]
    closed = get_pull_requests(PullRequestInput(**REPO, state="closed")).pull_requests
    assert [(p.number, p.state) for p in closed] == [(11, "closed")]
    everything = get_pull_requests(PullRequestInput(**REPO, state="all")).pull_requests
    assert [p.state for p in everything] == ["merged", "closed"]


def test_commits_truncate_message_and_limit(gh):
    result = get_commits(CommitInput(**REPO, limit=3))
    assert len(result.commits) == 3 and len(result.commits[0].message) == 300


def test_commits_empty_repo(gh):
    gh.overrides["/repos/octo/demo/commits"] = httpx.Response(409, json={"message": "Git Repository is empty."})
    assert get_commits(CommitInput(**REPO)).commits == []


# -- search --------------------------------------------------------------------

def test_search_requires_token(gh):
    assert search_code(SearchCodeInput(**REPO, query="main")).kind == "auth"


def test_search_with_token():
    fake = FakeGitHub()
    set_client(GitHubClient(token="t", transport=httpx.MockTransport(fake)))
    try:
        result = search_code(SearchCodeInput(**REPO, query="main"))
        assert result.matches[0].fragments == ["def main()"]
        assert "repo:octo/demo" in fake.calls[0].url.params["q"]
        assert fake.calls[0].headers["authorization"] == "Bearer t"
    finally:
        set_client(None)


# -- errors, retries, rate limits, caching ------------------------------------------

def test_not_found_is_structured_error(gh):
    result = get_repository(RepositoryInput(owner="octo", repo="missing"))
    assert is_error(result) and result.kind == "not_found" and result.status_code == 404


def test_rate_limit_error_and_short_circuit(gh):
    reset = int(time.time()) + 120
    gh.overrides["/repos/octo/demo"] = httpx.Response(
        403, json={"message": "API rate limit exceeded"},
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
    )
    result = get_repository(RepositoryInput(**REPO))
    assert result.kind == "rate_limited" and 100 <= result.retry_after_seconds <= 121
    assert "GITHUB_TOKEN" in result.message

    calls_before = len(gh.calls)
    again = get_repository(RepositoryInput(**REPO))
    assert again.kind == "rate_limited" and len(gh.calls) == calls_before  # no request made


def test_plain_403_is_forbidden(gh):
    gh.overrides["/repos/octo/demo"] = httpx.Response(403, json={"message": "Repository access blocked"})
    assert get_repository(RepositoryInput(**REPO)).kind == "forbidden"


def test_bad_token_is_auth_error(gh):
    gh.overrides["/repos/octo/demo"] = httpx.Response(401, json={"message": "Bad credentials"})
    assert get_repository(RepositoryInput(**REPO)).kind == "auth"


def test_retries_transient_server_errors():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(502)
        return FakeGitHub()(request)

    set_client(GitHubClient(transport=httpx.MockTransport(handler), sleep=lambda _: None))
    try:
        assert get_repository(RepositoryInput(**REPO)).name == "demo"
        assert len(attempts) == 3
    finally:
        set_client(None)


def test_network_failure_returns_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    set_client(GitHubClient(transport=httpx.MockTransport(handler), sleep=lambda _: None))
    try:
        assert get_repository(RepositoryInput(**REPO)).kind == "network"
    finally:
        set_client(None)


def test_unexpected_payload_does_not_raise(gh):
    gh.overrides["/repos/octo/demo"] = httpx.Response(200, json={"unexpected": True})
    result = get_repository(RepositoryInput(**REPO))
    assert result.kind == "unexpected"


def test_expired_cache_revalidates_with_etag(gh):
    client = GitHubClient(transport=httpx.MockTransport(gh), sleep=lambda _: None)
    first = client.get("/repos/octo/demo/contents/package.json", ttl=0)
    gh.overrides["/repos/octo/demo/contents/package.json"] = httpx.Response(304)
    second = client.get("/repos/octo/demo/contents/package.json", ttl=0)
    assert second == first
    assert gh.calls[-1].headers["if-none-match"] == '"v1"'


def test_disk_cache_survives_new_client(tmp_path, gh):
    from tools.cache import ResponseCache

    one = GitHubClient(cache=ResponseCache(tmp_path), transport=httpx.MockTransport(gh))
    one.get("/repos/octo/demo")
    calls = len(gh.calls)
    two = GitHubClient(cache=ResponseCache(tmp_path), transport=httpx.MockTransport(gh))
    assert two.get("/repos/octo/demo")["name"] == "demo"
    assert len(gh.calls) == calls


def test_invalid_input_rejected_by_schema():
    with pytest.raises(ValueError):
        RepositoryInput(owner="a/b", repo="x")
    with pytest.raises(ValueError):
        IssueInput(**REPO, state="bogus")


# -- accuracy helpers added after the first live specialist runs ---------------------------

def test_oldest_first_sorts_ascending(gh):
    get_issues(IssueInput(**REPO, oldest_first=True, limit=1))
    get_pull_requests(PullRequestInput(**REPO, oldest_first=True, state="merged"))
    issue_call = next(c for c in gh.calls if c.url.path.endswith("/issues"))
    pr_call = next(c for c in gh.calls if c.url.path.endswith("/pulls"))
    for call in (issue_call, pr_call):
        assert call.url.params["sort"] == "created" and call.url.params["direction"] == "asc"
    get_issues(IssueInput(**REPO))  # default is newest first: no sort params
    assert "direction" not in [c for c in gh.calls if c.url.path.endswith("/issues")][-1].url.params


def test_contributors_report_top_share_computed_in_code(gh):
    from tools import get_contributors
    from tools.schemas import ContributorInput

    gh.overrides["/repos/octo/demo/contributors"] = httpx.Response(200, json=[
        {"login": "a", "contributions": 60}, {"login": "b", "contributions": 30}, {"login": "c", "contributions": 10},
    ])
    result = get_contributors(ContributorInput(**REPO))
    assert result.total_contributions == 100 and result.top_contributor_share_pct == 60.0

    from tools.client import get_client

    get_client().cache.clear()  # otherwise the first response is served from cache
    gh.overrides["/repos/octo/demo/contributors"] = httpx.Response(204)
    empty = get_contributors(ContributorInput(**REPO))
    assert empty.contributors == [] and empty.top_contributor_share_pct == 0.0


def test_repository_exposes_combined_issue_and_pr_count_under_an_honest_name(gh):
    gh.overrides["/repos/octo/demo"] = httpx.Response(200, json={
        "name": "demo", "owner": {"login": "octo"}, "stargazers_count": 1, "default_branch": "main", "fork": False,
        "created_at": "t", "updated_at": "t", "open_issues_count": 85,
    })
    result = get_repository(RepositoryInput(**REPO))
    assert result.open_issues_and_prs == 85 and not hasattr(result, "open_issues")


# -- result shape: warnings survive a size cap because they come first -----------------------

def test_flags_come_before_bulky_data_so_a_cut_result_keeps_them():
    from tools.schemas import (
        ContributorsOutput, DependencyFile, FileContentOutput, IssuesOutput, PullRequestsOutput, RepositoryTreeOutput,
    )

    def fields(model):
        return list(model.model_fields)

    assert fields(FileContentOutput).index("truncated") < fields(FileContentOutput).index("content")
    assert fields(RepositoryTreeOutput).index("truncated") < fields(RepositoryTreeOutput).index("tree")
    assert fields(IssuesOutput)[:3] == ["note", "returned", "has_more"] and fields(IssuesOutput)[-1] == "issues"
    assert fields(PullRequestsOutput)[-1] == "pull_requests"
    assert fields(ContributorsOutput)[-1] == "contributors"
    assert fields(DependencyFile).index("truncated") < fields(DependencyFile).index("content")


def test_truncated_flag_is_visible_at_the_start_of_the_json(gh):
    result = get_file_content(FileContentInput(**REPO, path="README.md", max_chars=100))
    assert result.truncated
    assert result.model_dump_json().index('"truncated":true') < result.model_dump_json().index('"content"')


def test_capped_lists_carry_a_note_that_the_count_is_not_a_total(gh):
    capped = get_issues(IssueInput(**REPO, limit=1))
    assert capped.has_more and capped.returned == 1 and "MORE than 1" in capped.note
    complete = get_issues(IssueInput(**REPO))
    assert not complete.has_more and "complete list" in complete.note and "exact total" in complete.note
    prs = get_pull_requests(PullRequestInput(**REPO, state="merged", limit=1))
    assert prs.has_more is False or "MORE than" in prs.note


def test_tree_nodes_carry_no_sha(gh):
    result = get_repository_tree(RepositoryTreeInput(**REPO))
    assert all(not hasattr(node, "sha") for node in result.tree)
    assert "sha" not in result.model_dump_json()


# -- every list says whether it is complete, and wastes no tokens on things the model never uses --------

def test_commits_say_how_many_came_back_and_whether_that_is_all(gh):
    capped = get_commits(CommitInput(**REPO, limit=3))
    # the fake serves limit+1 items, so a capped list is detected: 3 returned, more exist
    assert capped.returned == 3 and capped.has_more is True and "MORE than 3" in capped.note
    assert len(capped.commits) == 3


def test_releases_are_slim_and_carry_a_note(gh):
    from tools import get_releases
    from tools.schemas import ReleaseInput

    gh.overrides["/repos/octo/demo/releases"] = httpx.Response(200, json=[
        {"tag_name": "v2", "name": "Two", "published_at": "2026-01-01", "body": "x" * 5000,
         "assets": [{"name": "a.whl", "size": 1, "download_count": 9}, {"name": "b.tar.gz", "size": 2, "download_count": 3}]},
    ])
    result = get_releases(ReleaseInput(**REPO))
    [release] = result.releases
    assert release.asset_count == 2 and len(release.body) == 300
    assert "assets" not in result.model_dump_json() and "download_count" not in result.model_dump_json()
    assert result.returned == 1 and "complete list" in result.note


def test_contributors_carry_no_avatar_urls_and_report_their_own_count(gh):
    from tools import get_contributors
    from tools.schemas import ContributorInput

    gh.overrides["/repos/octo/demo/contributors"] = httpx.Response(200, json=[
        {"login": "a", "contributions": 5, "avatar_url": "https://example.test/a.png"},
        {"login": "b", "contributions": 5, "avatar_url": "https://example.test/b.png"},
    ])
    result = get_contributors(ContributorInput(**REPO))
    assert result.returned == 2 and "avatar" not in result.model_dump_json()
    assert result.top_contributor_share_pct == 50.0


# -- search_cve (Tavily-backed CVE lookup) --------------------------------------

def test_search_cve_requires_api_key(monkeypatch):
    from tools import search_cve
    from tools.schemas import SearchCVEInput

    # setenv to "" here, NOT delenv: dotenv only fills in a var that is absent from os.environ, so
    # delenv would let `search_cve`'s own load_dotenv() reload a real key from a developer's .env
    # (see tests/conftest.py's `normal_limits_by_default` for the same gotcha with LITE_MODE).
    monkeypatch.setenv("TAVILY_API_KEY", "")
    result = search_cve(SearchCVEInput(package="lodash"))
    assert is_error(result) and result.kind == "auth" and "TAVILY_API_KEY" in result.message


def test_search_cve_happy_path_extracts_cve_ids(monkeypatch):
    from tools import search_cve
    from tools.cve import set_tavily_client
    from tools.schemas import SearchCVEInput

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    requests = []

    def fake_tavily(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": [
            {"title": "CVE-2023-12345 in lodash", "content": "Prototype pollution in lodash before 4.17.21.", "url": "https://nvd.nist.gov/vuln/detail/CVE-2023-12345"},
            {"title": "GHSA advisory", "content": "No CVE assigned yet.", "url": "https://github.com/advisories/GHSA-xxxx"},
        ]})

    set_tavily_client(httpx.Client(transport=httpx.MockTransport(fake_tavily)))
    try:
        result = search_cve(SearchCVEInput(package="lodash", version="4.17.15", ecosystem="npm"))
    finally:
        set_tavily_client(None)

    assert not is_error(result)
    assert len(result.matches) == 2
    assert result.matches[0].cve_id == "CVE-2023-12345" and result.matches[0].source == "nvd.nist.gov"
    assert result.matches[1].cve_id is None  # no CVE id in that result's text; must not be invented
    assert "lodash" in requests[0].content.decode() and "4.17.15" in requests[0].content.decode()


def test_search_cve_no_results_says_so_without_claiming_safety(monkeypatch):
    from tools import search_cve
    from tools.cve import set_tavily_client
    from tools.schemas import SearchCVEInput

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    set_tavily_client(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))))
    try:
        result = search_cve(SearchCVEInput(package="some-obscure-pkg"))
    finally:
        set_tavily_client(None)
    assert result.matches == [] and "does not mean it has no known CVEs" in result.note


def test_search_cve_budget_is_capped_per_toolbox_run(monkeypatch):
    """The Security Specialist's toolbox enforces a call budget so the model cannot spend unlimited Tavily calls."""
    from agents.limits import LITE
    from agents.toolbox import Toolbox
    from memory import MemoryStore

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    from tools.cve import set_tavily_client
    set_tavily_client(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))))
    store = MemoryStore(":memory:")
    try:
        box = Toolbox("octo", "demo", store, "s1", agent_name="security_specialist",
                       include=("search_cve",), limits=LITE)  # LITE caps max_cve_lookups at 3
        # distinct args each call so the toolbox's identical-call dedup doesn't short-circuit before the budget check
        outcomes = [box.call("search_cve", f'{{"package": "pkg-{i}"}}') for i in range(LITE.max_cve_lookups + 2)]
    finally:
        set_tavily_client(None)
        store.close()

    assert sum(o.is_error for o in outcomes) == 2  # the last 2 calls exceed the budget
    assert "budget used up" in outcomes[-1].content
