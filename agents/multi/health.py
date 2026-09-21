"""
Project Health Specialist agent.

Analyzes project activity and health metrics: commits, issues, PRs,
release cadence, contributor distribution, and staleness indicators.

Findings written to memory (categories):
- activity, maintenance, issue_management, releases, community, bus_factor
"""

from agents.multi.specialist import SpecialistAgent
from prompts.multi_agent import get_health_prompt


class HealthSpecialist(SpecialistAgent):
    """
    Specialist agent for project health analysis.

    Investigates activity, maintenance status, and community metrics.
    """

    agent_id = "health_specialist"
    title = "Project Health"
    tools = ("get_repository", "get_commits", "get_issues", "get_pull_requests", "get_releases", "get_contributors")
    prompt = staticmethod(get_health_prompt)
