"""
Live check for the War Room specialists (needs OPENAI_API_KEY; GITHUB_TOKEN recommended).

Usage:
    python scripts/demo_specialists.py pallets/click            # all four, in order
    python scripts/demo_specialists.py pallets/click health     # just one: architecture|health|quality|security

Findings go into a throwaway database (demo_specialists.db), not your real agent_memory.db.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.multi import SPECIALISTS  # noqa: E402
from memory import MemoryStore  # noqa: E402
from tools import parse_repo_ref  # noqa: E402

if __name__ == "__main__":
    ref = parse_repo_ref(sys.argv[1] if len(sys.argv) > 1 else "pallets/click")
    only = sys.argv[2] if len(sys.argv) > 2 else None
    store = MemoryStore("demo_specialists.db")
    session = store.create_session(f"{ref.owner}/{ref.repo}", "multi_agent")

    for cls in SPECIALISTS:
        if only and only not in cls.agent_id:
            continue
        print(f"\n=== {cls.title} specialist ===")
        agent = cls(store)
        for event in agent.run(ref.owner, ref.repo, session):
            if event.kind == "reasoning":
                print(f"  💭 {event.content}")
            elif event.kind == "tool_result":
                print(f"  🔧 {event.name} {event.arguments} -> {event.content}")
            elif event.kind in ("final", "error"):
                print(f"  {'✅' if event.kind == 'final' else '❌'} {event.content}")
        for f in store.get_findings(f"{ref.owner}/{ref.repo}", agent=cls.agent_id, session_id=session)[::-1]:
            print(f"  [{f.severity}/{f.category}] {f.finding}")
