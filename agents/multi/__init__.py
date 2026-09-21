"""
Multi-agent specialists for war room mode.

Implements four specialized agents (Architecture, Security, Code Quality, Project Health)
that work in parallel or sequence, writing structured findings to shared memory.
All specialists are coordinated by a Manager agent that synthesizes the final report.
"""

from agents.multi.architecture import ArchitectureSpecialist
from agents.multi.health import HealthSpecialist
from agents.multi.quality import QualitySpecialist
from agents.multi.security import SecuritySpecialist
from agents.multi.specialist import SpecialistAgent, SpecialistResult

# Recommended run order: Security last, because it benefits from what the others learned
# (e.g. the tech stack) via shared memory.
SPECIALISTS = (ArchitectureSpecialist, HealthSpecialist, QualitySpecialist, SecuritySpecialist)

__all__ = [
    "ArchitectureSpecialist",
    "HealthSpecialist",
    "QualitySpecialist",
    "SecuritySpecialist",
    "SpecialistAgent",
    "SpecialistResult",
    "SPECIALISTS",
]
