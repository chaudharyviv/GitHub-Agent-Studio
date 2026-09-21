"""
GitHub tools module for GitHub Agent Studio.

This module provides clean, typed interfaces to GitHub repository data.
Every tool takes a Pydantic input model and returns a Pydantic output model,
or a ``ToolError`` if something went wrong (tools never raise).
"""

from tools.github import (
    get_commits,
    get_contributors,
    get_dependency_files,
    get_file_content,
    get_issues,
    get_pull_requests,
    get_releases,
    get_repository,
    get_repository_tree,
    search_code,
)
from tools.schemas import ToolError, is_error, parse_repo_ref

__all__ = [
    "get_repository",
    "get_repository_tree",
    "get_file_content",
    "get_dependency_files",
    "get_issues",
    "get_pull_requests",
    "get_commits",
    "get_releases",
    "get_contributors",
    "search_code",
    "ToolError",
    "is_error",
    "parse_repo_ref",
]
