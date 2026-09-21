"""
Live smoke test for the GitHub tools layer (Phase 1 acceptance check).

Usage:
    python scripts/demo_tools.py                       # three default repos
    python scripts/demo_tools.py owner/repo [...]      # your own repos

Uses GITHUB_TOKEN from the environment / .env if present. Without a token you
get 60 requests/hour; each repo costs roughly 10 requests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (  # noqa: E402
    get_commits,
    get_contributors,
    get_dependency_files,
    get_file_content,
    get_issues,
    get_pull_requests,
    get_releases,
    get_repository,
    get_repository_tree,
    is_error,
    parse_repo_ref,
)
from tools.client import get_client  # noqa: E402
from tools.schemas import (  # noqa: E402
    CommitInput,
    ContributorInput,
    DependencyFilesInput,
    FileContentInput,
    IssueInput,
    PullRequestInput,
    ReleaseInput,
    RepositoryInput,
    RepositoryTreeInput,
)

DEFAULT_REPOS = ["octocat/Hello-World", "pallets/click", "langchain-ai/langchain"]


def show(label: str, result, summary) -> bool:
    if is_error(result):
        print(f"  [FAIL] {label}: {result.kind} - {result.message}")
        return False
    print(f"  [ ok ] {label}: {summary(result)}")
    return True


def run(ref: str) -> bool:
    print(f"\n=== {ref} ===")
    r = parse_repo_ref(ref)
    ident = {"owner": r.owner, "repo": r.repo}
    ok = True

    repo = get_repository(RepositoryInput(**ident))
    ok &= show("get_repository", repo, lambda x: f"{x.stars:,} stars, {x.language}, branch {x.default_branch}, license {x.license}")
    if is_error(repo):
        return False

    tree = get_repository_tree(RepositoryTreeInput(**ident))
    ok &= show("get_repository_tree", tree, lambda x: f"{len(x.tree)} root entries, truncated={x.truncated}")

    readme = next((n.path for n in getattr(tree, "tree", []) if n.path.lower().startswith("readme")), None)
    if readme:
        ok &= show("get_file_content", get_file_content(FileContentInput(**ident, path=readme, max_chars=2000)),
                   lambda x: f"{x.path}: {x.size:,} bytes, truncated={x.truncated}")

    ok &= show("get_dependency_files", get_dependency_files(DependencyFilesInput(**ident)),
               lambda x: f"{[f.path for f in x.files]} -> {x.detected_languages}")
    ok &= show("get_issues", get_issues(IssueInput(**ident, limit=10)), lambda x: f"{x.total_count} open, has_more={x.has_more}")
    ok &= show("get_pull_requests", get_pull_requests(PullRequestInput(**ident, state="merged", limit=10)),
               lambda x: f"{x.total_count} merged, has_more={x.has_more}")
    ok &= show("get_commits", get_commits(CommitInput(**ident, limit=5)),
               lambda x: f"{len(x.commits)} commits, latest {x.commits[0].timestamp if x.commits else 'n/a'}")
    ok &= show("get_releases", get_releases(ReleaseInput(**ident, limit=5)),
               lambda x: f"{len(x.releases)} releases, latest {x.releases[0].tag_name if x.releases else 'n/a'}")
    ok &= show("get_contributors", get_contributors(ContributorInput(**ident, limit=5)),
               lambda x: ", ".join(f"{c.login} ({c.contributions})" for c in x.contributors))
    return bool(ok)


if __name__ == "__main__":
    repos = sys.argv[1:] or DEFAULT_REPOS
    results = [run(ref) for ref in repos]
    for resource, limit in get_client().rate_limit_status().items():
        print(f"\nrate limit [{resource}]: {limit.remaining} requests remaining")
    sys.exit(0 if all(results) else 1)
