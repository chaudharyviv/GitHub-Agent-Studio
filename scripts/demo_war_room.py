"""
Run the whole War Room from the command line: specialists in order, then the Manager's report.

Usage:
    python scripts/demo_war_room.py pallets/click                 # all four specialists + Manager
    python scripts/demo_war_room.py pallets/click health security  # only these specialists + Manager

Cheaper test runs: set LITE_MODE=1 (PowerShell: $env:LITE_MODE=1). Needs OPENAI_API_KEY; GITHUB_TOKEN recommended.
Everything goes into a throwaway database (demo_war_room.db), not your real agent_memory.db.
The finished report is also written to demo_war_room_report.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles and redirected output default to a legacy encoding that cannot print the emoji below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from agents.limits import get_limits, lite_mode_enabled  # noqa: E402
from memory import MemoryStore  # noqa: E402
from orchestration.runner import stream_war_room  # noqa: E402
from tools import parse_repo_ref  # noqa: E402

if __name__ == "__main__":
    ref = parse_repo_ref(sys.argv[1] if len(sys.argv) > 1 else "pallets/click")
    only = sys.argv[2:] or None
    print(f"Mode: {'LITE' if lite_mode_enabled() else 'normal'} ({get_limits().specialist_steps} tool rounds per specialist)")

    report = None
    for ev in stream_war_room(ref.owner, ref.repo, store=MemoryStore("demo_war_room.db"), only=only):
        if ev.kind == "error":
            sys.exit(f"ERROR: {ev.message}")
        elif ev.kind == "agent_start":
            print(f"\n=== {ev.title} specialist ===")
        elif ev.kind == "agent":
            e = ev.event
            if e.kind == "tool_result":
                print(f"  🔧 {e.name} {e.arguments} -> {e.content}")
            elif e.kind == "memory":
                print(f"  🧠 {e.content}")
            elif e.kind == "error":
                print(f"  ❌ {e.content}")
        elif ev.kind == "agent_done":
            status = next(s for s in ev.statuses if s.agent_id == ev.agent_id)
            print(f"  → {status.state}, {status.findings} finding(s) · 💲 {ev.usage}")
        elif ev.kind == "manager_start":
            print("\n=== Manager writing the report ===")
        elif ev.kind == "report":
            report = ev.report
        elif ev.kind == "done":
            print(f"\nTOTAL: {ev.usage}")

    if report:
        Path("demo_war_room_report.md").write_text(report.markdown, encoding="utf-8")
        print("\n" + "=" * 78 + "\n")
        print(report.markdown)
        if report.unknown_refs:
            print(f"\n⚠️ The narrative cited unknown finding ids: {report.unknown_refs}")
