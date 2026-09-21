"""
Security Specialist agent.

Static, heuristic security review: dependency manifests, security policy,
CI workflow risks, committed secrets and risky configuration. It is not a
vulnerability scanner and never claims a CVE it has not been shown.

Findings written to memory (categories):
- security_policy, dependency_risk, secrets, ci_security, configuration, hardening
"""

from agents.multi.specialist import SpecialistAgent
from prompts.multi_agent import get_security_prompt


class SecuritySpecialist(SpecialistAgent):
    """
    Specialist agent for security analysis.

    Investigates dependency risks, security policies, and anti-patterns.
    """

    agent_id = "security_specialist"
    title = "Security"
    tools = ("get_repository_tree", "get_file_content", "get_dependency_files", "search_code")
    prompt = staticmethod(get_security_prompt)
