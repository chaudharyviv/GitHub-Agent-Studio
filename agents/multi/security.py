"""
Security Specialist agent.

Static, heuristic security review: dependency manifests, security policy,
CI workflow risks, committed secrets and risky configuration, plus an optional
live CVE lookup (search_cve, Tavily-backed, biased toward NVD / GitHub
Advisories) for dependencies it flags. Still not a real vulnerability scanner
or SCA pipeline, and it never claims a CVE id that search_cve did not return.

Findings written to memory (categories):
- security_policy, dependency_risk, secrets, ci_security, configuration, hardening
"""

from agents.multi.specialist import SpecialistAgent
from prompts.multi_agent import SECURITY_CATEGORIES, get_security_prompt


class SecuritySpecialist(SpecialistAgent):
    """
    Specialist agent for security analysis.

    Investigates dependency risks, security policies, and anti-patterns.
    """

    agent_id = "security_specialist"
    title = "Security"
    tools = ("get_repository_tree", "get_file_content", "get_dependency_files", "search_code", "search_cve")
    categories = SECURITY_CATEGORIES
    prompt = staticmethod(get_security_prompt)
