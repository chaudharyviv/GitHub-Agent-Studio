"""
Architecture Specialist agent.

Analyzes repository structure, modules, entry points, design patterns,
and dependency relationships to understand the codebase architecture.

Findings written to memory (categories):
- structure: how the repository is organized
- entry_points: where execution starts
- modules: key modules and their responsibilities
- patterns: design/architectural patterns identified
- tech_stack: frameworks and major libraries
"""

from agents.multi.specialist import SpecialistAgent
from prompts.multi_agent import ARCHITECTURE_CATEGORIES, get_architecture_prompt


class ArchitectureSpecialist(SpecialistAgent):
    """
    Specialist agent for architecture analysis.

    Investigates repository structure, modularity, and design patterns.
    """

    agent_id = "architecture_specialist"
    title = "Architecture"
    tools = ("get_repository", "get_repository_tree", "get_file_content", "get_dependency_files")
    categories = ARCHITECTURE_CATEGORIES
    prompt = staticmethod(get_architecture_prompt)
