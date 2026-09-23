"""
Tavily-backed CVE / vulnerability lookup for the Security Specialist.

One tool, ``search_cve``: live web search biased toward authoritative sources (NVD,
GitHub Advisories, OSV, Snyk), turned into a small structured result. This is not a
real SCA pipeline or vulnerability database match — no dependency graph, no verified
CVE feed, just what a web search turns up. Requires TAVILY_API_KEY; without it, the
tool returns a clear "unavailable" error the model can reason about instead of
inventing CVE ids.
"""

import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from tools.schemas import CVEMatch, SearchCVEInput, SearchCVEOutput, ToolError

TAVILY_URL = "https://api.tavily.com/search"
_CVE_ID_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_PREFERRED_DOMAINS = ["nvd.nist.gov", "github.com", "osv.dev", "snyk.io"]


def _build_query(input_data: SearchCVEInput) -> str:
    parts = [input_data.package]
    if input_data.ecosystem:
        parts.append(input_data.ecosystem)
    if input_data.version:
        parts.append(input_data.version)
    parts.append("CVE vulnerability")
    return " ".join(parts)


def _first_cve_id(text: str) -> Optional[str]:
    match = _CVE_ID_RE.search(text)
    return match.group(0).upper() if match else None


def search_cve(input_data: SearchCVEInput) -> SearchCVEOutput | ToolError:
    """
    Look up known CVEs for a package by live web search, biased toward NVD and GitHub Advisories.

    Not a real vulnerability database: it reports what a web search turns up, so a CVE id
    is only as reliable as the source page. Requires TAVILY_API_KEY.
    """
    api_key = _tavily_api_key()
    if not api_key:
        return ToolError(
            kind="auth",
            message="CVE lookup unavailable: TAVILY_API_KEY is not set. Rely on dependency-file inspection only; never invent a CVE id.",
        )

    try:
        response = _client().post(TAVILY_URL, json={
            "api_key": api_key,
            "query": _build_query(input_data),
            "search_depth": "basic",
            "max_results": 5,
            "include_domains": _PREFERRED_DOMAINS,
        })
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return ToolError(kind="api_error", message=f"Tavily search failed: HTTP {exc.response.status_code}")
    except httpx.TransportError as exc:
        return ToolError(kind="network", message=f"Could not reach Tavily: {exc.__class__.__name__}: {exc}")

    matches = [
        CVEMatch(
            cve_id=_first_cve_id(f"{item.get('title', '')} {item.get('content', '')}"),
            summary=(item.get("content") or "")[:500],
            source_url=item.get("url", ""),
            source=urlparse(item.get("url", "")).netloc or "unknown",
        )
        for item in data.get("results", []) if item.get("url")
    ]
    note = (
        f"{len(matches)} search result(s): leads from a live web search, not verified database matches. "
        "Cite only cve_id values actually present here; never invent one."
        if matches else
        "No search results found for this package/version. That does not mean it has no known CVEs, only that this search did not surface one."
    )
    return SearchCVEOutput(package=input_data.package, version=input_data.version, matches=matches, note=note)


def _tavily_api_key() -> Optional[str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("TAVILY_API_KEY") or None


# ---------------------------------------------------------------------------
# Shared default HTTP client (swappable in tests, same pattern as tools.client)
# ---------------------------------------------------------------------------

_default_client: Optional[httpx.Client] = None


def _client() -> httpx.Client:
    global _default_client
    if _default_client is None:
        _default_client = httpx.Client(timeout=15.0)
    return _default_client


def set_tavily_client(client: Optional[httpx.Client]) -> None:
    """Replace the shared HTTP client (used by tests); None resets to lazy default."""
    global _default_client
    _default_client = client
