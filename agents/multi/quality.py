"""
Code Quality Specialist agent.

Analyzes code quality signals: file sizes, complexity heuristics,
test presence, documentation, tooling and error handling patterns.

Findings written to memory (categories):
- testing, documentation, complexity, tooling, error_handling, organization
"""

from agents.multi.specialist import SpecialistAgent
from prompts.multi_agent import QUALITY_CATEGORIES, get_quality_prompt


class QualitySpecialist(SpecialistAgent):
    """
    Specialist agent for code quality analysis.

    Investigates code organization, tests, tooling, and documentation.
    """

    agent_id = "quality_specialist"
    title = "Code Quality"
    tools = ("get_repository", "get_repository_tree", "get_file_content", "get_dependency_files")
    categories = QUALITY_CATEGORIES
    prompt = staticmethod(get_quality_prompt)
