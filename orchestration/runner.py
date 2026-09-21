"""
Agent orchestration and execution runners.

This module provides two main entry points:
- run_single_agent: Execute single transparent agent mode
- run_multi_agent: Coordinate war room specialists + manager
"""

from typing import Any, Optional


def run_single_agent(
    owner: str,
    repo: str,
    user_query: Optional[str] = None,
) -> Any:
    """
    Execute a single-agent investigation into a repository.
    
    Runs the transparent single agent with full tool-calling visibility.
    
    The agent will:
    1. Load previous findings from memory (if any)
    2. Call tools based on LLM decisions
    3. Write new findings to memory
    4. Return structured investigation results
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        user_query: Optional user directive (e.g., "Analyze", "Teach me")
        
    Returns:
        Investigation results with visible tool calls and findings
        
    Raises:
        NotImplementedError: Pending implementation in Phase 3
    """
    raise NotImplementedError("run_single_agent will be implemented in Phase 3: Single Agent Mode")


def run_multi_agent(
    owner: str,
    repo: str,
    user_query: Optional[str] = None,
) -> Any:
    """
    Execute multi-agent war room investigation into a repository.
    
    Orchestrates four specialist agents (Architecture, Security, Quality, Health)
    and a Manager agent that synthesizes the findings into a final report.
    
    Process:
    1. Specialists are run sequentially or in limited parallelism
    2. Each specialist writes structured findings to shared memory
    3. Manager reads all findings and produces a final health report
    4. All reasoning and tool calls are logged for inspection
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        user_query: Optional additional context for the investigation
        
    Returns:
        Comprehensive repository health report with specialist findings
        
    Raises:
        NotImplementedError: Pending implementation in Phase 5
    """
    raise NotImplementedError("run_multi_agent will be implemented in Phase 5: Manager + Orchestration")
