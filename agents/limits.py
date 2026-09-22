"""
Size limits that control how many tokens (and so how much money) an investigation costs.

The main cost driver is *input* tokens: every tool result stays in the conversation and is
re-sent to the model on each later round, so cost grows with (result size) x (rounds).
Set ``LITE_MODE=1`` (env or .env) while testing to shrink both.

    NORMAL: thorough investigations
    LITE:   roughly a third of the context, for cheap test runs
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    tool_result_chars: int  # cap on what any one tool result adds to the conversation
    file_chars: int  # characters read from one file
    dependency_file_chars: int  # characters read from each dependency manifest
    tree_entries: int  # entries returned by one repository-tree call
    list_limit: int  # most items any list tool (issues, PRs, commits, ...) may return
    single_agent_steps: int  # tool-calling rounds for the single agent
    specialist_steps: int  # tool-calling rounds for each specialist


# tool_result_chars must stay comfortably above file_chars: JSON escaping (newlines, quotes) inflates text,
# and a result cut by this cap loses its trailing fields. It is a safety net, not the normal way results get short.
NORMAL = Limits(tool_result_chars=20_000, file_chars=12_000, dependency_file_chars=6_000,
                tree_entries=400, list_limit=100, single_agent_steps=10, specialist_steps=8)
LITE = Limits(tool_result_chars=7_000, file_chars=4_500, dependency_file_chars=2_500,
              tree_entries=100, list_limit=12, single_agent_steps=6, specialist_steps=6)


def lite_mode_enabled() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("LITE_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def get_limits() -> Limits:
    return LITE if lite_mode_enabled() else NORMAL
