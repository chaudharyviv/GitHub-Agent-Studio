"""
Thin GitHub REST client: auth, caching, retries and rate-limit handling.

Everything that can go wrong is raised as ``GitHubAPIError``, which knows how to
turn itself into a ``ToolError``. The tool functions never let it escape.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from tools.cache import CacheEntry, ResponseCache
from tools.schemas import ErrorKind, ToolError

API_ROOT = "https://api.github.com"
DEFAULT_TTL = 300.0  # seconds a cached response is served without revalidation


class GitHubAPIError(Exception):
    """A failed GitHub call, carrying enough detail to build a ToolError."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        status_code: Optional[int] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds

    def to_tool_error(self) -> ToolError:
        return ToolError(
            kind=self.kind,
            message=self.message,
            status_code=self.status_code,
            retry_after_seconds=self.retry_after_seconds,
        )


@dataclass
class RateLimit:
    remaining: int
    reset_at: float  # epoch seconds


class GitHubClient:
    def __init__(
        self,
        token: Optional[str] = None,
        *,
        cache: Optional[ResponseCache] = None,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.token = token or None
        self.cache = cache if cache is not None else ResponseCache()
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._limits: dict[str, RateLimit] = {}  # keyed by GitHub rate-limit resource ("core", "search")
        self._http = httpx.Client(base_url=API_ROOT, timeout=timeout, transport=transport)

    def rate_limit_status(self) -> dict[str, RateLimit]:
        """Last rate-limit state seen per resource, for display in the UI."""
        return dict(self._limits)

    def get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        *,
        ttl: float = DEFAULT_TTL,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        """GET a JSON endpoint, using the cache. Returns parsed JSON (or None for 204)."""
        key = self._cache_key(path, params, accept)
        cached = self.cache.get(key)
        if cached is not None and cached.is_fresh(ttl):
            return cached.data

        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if cached is not None and cached.etag:
            headers["If-None-Match"] = cached.etag

        self._check_known_limit(path)
        response = self._request_with_retries(path, params, headers)
        self._record_rate_limit(response)

        if response.status_code == 304 and cached is not None:
            self.cache.put(key, CacheEntry(cached.data, cached.etag, time.time()))
            return cached.data
        if response.status_code >= 400:
            raise self._error_for(response)

        data = None if response.status_code == 204 else response.json()
        self.cache.put(key, CacheEntry(data, response.headers.get("etag"), time.time()))
        return data

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _cache_key(path: str, params: Optional[dict[str, Any]], accept: str) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        return f"{accept}|{path}?{query}"

    def _request_with_retries(self, path: str, params: Optional[dict[str, Any]], headers: dict) -> httpx.Response:
        last_error: Optional[GitHubAPIError] = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(self._backoff * 2 ** (attempt - 1))
            try:
                response = self._http.get(path, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_error = GitHubAPIError("network", f"Could not reach GitHub: {exc.__class__.__name__}: {exc}")
                continue
            if response.status_code >= 500:
                last_error = GitHubAPIError(
                    "api_error", f"GitHub returned a server error ({response.status_code}).", response.status_code
                )
                continue
            return response
        assert last_error is not None
        raise last_error

    def _record_rate_limit(self, response: httpx.Response) -> None:
        try:
            resource = response.headers.get("x-ratelimit-resource", "core")
            self._limits[resource] = RateLimit(
                remaining=int(response.headers["x-ratelimit-remaining"]),
                reset_at=float(response.headers["x-ratelimit-reset"]),
            )
        except (KeyError, ValueError):
            pass

    def _check_known_limit(self, path: str) -> None:
        """Skip the request entirely if we already know the bucket is empty."""
        limit = self._limits.get("search" if path.startswith("/search") else "core")
        if limit and limit.remaining <= 0 and limit.reset_at > time.time():
            raise self._rate_limited(int(limit.reset_at - time.time()) + 1)

    def _rate_limited(self, wait: int, status: Optional[int] = None) -> GitHubAPIError:
        hint = "" if self.token else " Set GITHUB_TOKEN to raise the limit from 60 to 5000 requests/hour."
        return GitHubAPIError(
            "rate_limited",
            f"GitHub API rate limit reached. Retry in about {wait}s.{hint}",
            status,
            retry_after_seconds=wait,
        )

    def _error_for(self, response: httpx.Response) -> GitHubAPIError:
        status = response.status_code
        try:
            detail = response.json().get("message", "")
        except ValueError:
            detail = ""

        if status in (403, 429):
            retry_after = response.headers.get("retry-after")
            exhausted = response.headers.get("x-ratelimit-remaining") == "0"
            if retry_after or exhausted or "rate limit" in detail.lower():
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    reset = float(response.headers.get("x-ratelimit-reset", time.time() + 60))
                    wait = max(int(reset - time.time()), 1)
                return self._rate_limited(wait, status)
            return GitHubAPIError("forbidden", f"Access forbidden: {detail or 'no details'}", status)
        if status == 401:
            return GitHubAPIError("auth", "GitHub rejected the token. Check that GITHUB_TOKEN is valid and unexpired.", status)
        if status == 404:
            return GitHubAPIError("not_found", "Not found. The repository or path does not exist, or is private.", status)
        if status == 409:
            return GitHubAPIError("empty_repository", f"Repository is empty or in a conflicting state: {detail}", status)
        if status == 451:
            return GitHubAPIError("forbidden", "This repository is unavailable for legal reasons.", status)
        if status == 422:
            return GitHubAPIError("invalid_input", f"GitHub rejected the request: {detail}", status)
        return GitHubAPIError("api_error", f"GitHub API error {status}: {detail}", status)


# ---------------------------------------------------------------------------
# Shared default client
# ---------------------------------------------------------------------------

_default_client: Optional[GitHubClient] = None


def get_client() -> GitHubClient:
    """Process-wide client, built lazily from GITHUB_TOKEN / GITHUB_CACHE_DIR."""
    global _default_client
    if _default_client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        cache_dir = os.environ.get("GITHUB_CACHE_DIR")
        _default_client = GitHubClient(
            token=os.environ.get("GITHUB_TOKEN"),
            cache=ResponseCache(cache_dir) if cache_dir else None,
        )
    return _default_client


def set_client(client: Optional[GitHubClient]) -> None:
    """Replace the shared client (used by tests); None resets to lazy default."""
    global _default_client
    _default_client = client
