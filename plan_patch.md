# plan_patch.md — GitHub Agent Studio Evolution

**Goal:** Turn the current educational single/multi-agent demo into a stronger teaching artifact and portfolio piece while keeping transparency and low cost.

**Target environment:** Local development + Streamlit Cloud deployment.

**Guiding principles**
- Keep the custom tool-calling loop (no framework takeover).
- Make every non-obvious design decision visible and documented.
- Prefer structured data over free-text wherever agents communicate.
- Default to cheap / fast behaviour on Streamlit Cloud (Lite mode).
- Leave clear extension points for multi-provider LLMs and persistent storage.

---

## Phase 0 — Housekeeping (do first)

### 0.1 Fix README clone URL
- Change the clone example from `your-org/github-agent-studio` to the real repository URL.
- File: `README.md`

### 0.2 Add “Limitations of the analysis” block to every War Room report
- Always append a short, fixed limitations section at the end of the generated markdown report.
- Suggested text (pre-CVE tool):

```markdown
## Limitations of this analysis
- Public repositories only.
- Security findings are heuristic (dependency-file inspection + model reasoning). This is **not** a real vulnerability scan or SCA.
- No full static analysis, license-compliance engine, or secret scanning.
- Analysis depth is limited by step budgets, result-size caps, and model context.
- Results depend on the capabilities of the configured model (`gpt-4o-mini` by default) and the system prompts.
```

- After Phase 1.5 (Tavily CVE tool) ships, update the security bullet to:

```markdown
- Security findings combine dependency-file inspection with live CVE lookup (Tavily → NVD / GitHub Advisories). This is still not a full SCA pipeline or exploit verification.
```

- Location: `agents/multi/report.py` (inside `assemble_report` or equivalent).
- Also surface the same short note in the Streamlit War Room UI under the report.

### 0.3 Update Technical Decisions Log
Replace / extend the existing table with:

```markdown
## 3. Technical Decisions Log

| Decision              | Choice                                      | Alternatives Considered              | Why                                                                 |
|-----------------------|---------------------------------------------|--------------------------------------|---------------------------------------------------------------------|
| UI framework          | Streamlit                                   | Next.js + Vercel AI SDK              | Faster educational prototype; pure Python                           |
| Agent framework       | Custom loop                                 | LangGraph, CrewAI                    | Transparency & teaching value                                       |
| Multi-agent execution | Parallel specialists + Manager merge        | Pure sequential                      | Realistic concurrency, lower latency, teaches merge & isolation     |
| Database              | SQLite (pluggable)                          | Supabase, pure JSON files, Postgres  | Simple & local for teaching; easy to swap for Cloud/production      |
| GitHub client         | `httpx` + manual + Pydantic                 | PyGithub                             | Lighter, full control over models                                   |
| LLM provider          | OpenAI (via thin `LLMProvider` abstraction) | Anthropic, local models, LiteLLM     | Best tool-calling support today; abstraction keeps multi-provider path clear |
| LLM model             | `gpt-4o-mini` (locked for v1)               | `gpt-4o`, `gpt-5.6`, etc.            | Best cost / speed / quality balance for frequent tool-calling loops |
| Caching               | Simple dict + optional disk                 | Redis, full HTTP cache               | Enough for demo scale                                               |
| Findings format       | Structured (Pydantic / tool args only)      | Free-text parsing by Manager         | Reliable, machine-readable, forces clean agent contracts            |
| Cost & size control   | Explicit `Limits` object                    | Hard-coded constants, no limits      | Makes cost/quality trade-offs visible and tunable for learners      |
| Vulnerability lookup  | Tavily → NVD / GitHub Advisories tool       | Offline CVE DB, full SCA engine, RAG | Targeted live lookup; upgrades Security Specialist without standing up retrieval infra |
```

---

## Phase 1 — Core cleanliness & educational clarity

### 1.1 Extract remaining magic numbers into `Limits` / settings

**File:** `agents/limits.py` (extend existing dataclass)

Add (or move) these fields:

```python
@dataclass(frozen=True)
class Limits:
    # existing
    tool_result_chars: int
    file_chars: int
    dependency_file_chars: int
    tree_entries: int
    list_limit: int
    single_agent_steps: int
    specialist_steps: int

    # newly extracted
    history_limit: int = 20
    max_findings_for_manager: int = 80
    max_evidence_chars: int = 200
    max_manifest_depth: int = 4
    max_manifests: int = 12
    tree_ttl: float = 600.0
    content_ttl: float = 900.0
    activity_ttl: float = 120.0
    max_commit_message_chars: int = 300
    max_release_body_chars: int = 300
    max_reasoning_chars: int = 800          # optional prose guard
```

**Actions**
- Replace every hard-coded constant in:
  - `agents/toolbox.py`
  - `agents/single.py`
  - `agents/multi/manager.py`
  - `tools/github.py`
  - `tools/client.py` / cache TTLs
- Keep `NORMAL` and `LITE` presets; Lite continues to shrink the size-related fields.
- Document the rationale for each limit in a short comment or in `spec.md`.

### 1.2 Injectable LLM client + thin provider abstraction

**New module:** `llm/provider.py` (or `agents/llm.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """Return the raw provider response object (OpenAI-compatible shape preferred)."""
        ...

class OpenAIProvider(LLMProvider):
    def __init__(self, client: Any = None, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        ...
    def chat(self, messages, *, tools=None, max_tokens=2048, temperature=0.2):
        ...
```

**Changes**
- `resolve_llm()` now returns an `LLMProvider` instance.
- `SingleAgent`, every specialist, and `ManagerAgent` accept an optional `provider: LLMProvider`.
- `orchestration/runner.py` creates one shared provider and injects it.
- Streamlit sidebar / config can later expose a provider selector without touching agent loops.
- Keep the existing OpenAI-only path working with zero behaviour change.

**Educational value:** Learners can see exactly where a second provider would be plugged in.

### 1.3 Expand the Learning Path with concrete exercises

**File:** `README.md` (or new `LEARNING.md`)

Add a numbered exercise list, for example:

1. **Watch a single tool-calling loop**  
   Read `agents/loop.py` → `agents/single.py` → run Single Agent on `pallets/click` with a simple question. Observe live events.

2. **Memory side-effects**  
   Read `agents/toolbox.py` (`_save_finding`, `_after_call`). Ask a question that produces a finding, restart / clear chat, ask a follow-up that should recall it.

3. **Compare Single vs War Room**  
   Run the same repository through both modes and use the “Single vs Team” comparison tab.

4. **Limits & cost control**  
   Toggle Lite mode, change max-output-tokens, re-run, and compare token usage / report length.

5. **Parallel vs sequential (after Phase 2)**  
   Toggle the execution mode and observe the difference in wall-clock time and finding isolation.

Each exercise should name the exact files to open and a concrete query to type.

### 1.4 Structured findings only (no free-text reliance)

- Specialists already save findings via the `save_finding` tool (Pydantic-validated).
- Make the Manager consume **only** the structured findings list.
- Remove any remaining free-text parsing or “best-effort” extraction of findings from specialist prose.
- Keep the Manager’s narrative generation, but the evidence appendix and status table must be built purely from structured data.
- This change also makes the parallel merge (Phase 2) trivial and reliable.

### 1.5 Tavily-backed CVE / vulnerability lookup (Security Specialist upgrade)

**Why this, not a RAG pipeline**

Security is currently the weakest, most-flagged part of the report (the Limitations text admits “not a real vulnerability scan”). The right upgrade is a **targeted tool**, not a retrieval pipeline:

- There is no large unstructured corpus that needs embedding/search — the agents already pull files and READMEs directly via the GitHub API.
- Standing up embeddings + a vector store would solve a problem we do not have, purely to say “we have RAG.”
- If a RAG demo is desired later, it belongs as a separate track (e.g. “chat with this repo’s docs”), not bolted onto the War Room security flow.

**What to build (genuine quick win)**

One new tool:

```text
search_cve(package: str, version: str | None = None) → structured results
```

- Implementation: Tavily search, optionally biased/filtered toward `nvd.nist.gov`, GitHub Advisories, and similar authoritative sources.
- Returns a small Pydantic model (CVE id, severity, summary, source URL, matched package/version).
- Wired only into the **Security Specialist** toolbox (alongside existing dependency-file reads).
- The specialist can call it after `get_dependency_files` for packages it cares about.

**Demo value**

> “The Security Specialist now checks live CVE data instead of only pattern-matching dependency files.”

**Config / secrets**
- `TAVILY_API_KEY` in `.env` / Streamlit secrets (optional — if missing, the tool returns a clear “CVE lookup unavailable” error the model can reason about).
- Keep the tool cheap: hard limit on number of `search_cve` calls per specialist run (via `Limits` or a specialist-specific cap).

**Limitations text update**
Once this ships, soften the security bullet in the report Limitations block from “not a real vulnerability scan” to something like:

> Security findings combine dependency-file inspection with live CVE lookup (Tavily → NVD / GitHub Advisories). This is still not a full SCA pipeline or exploit verification.

**Effort:** S  
**Impact:** High (directly upgrades the most-criticised part of the analysis)

---

## Phase 2 — Parallel specialist execution (do now)

### 2.1 Parallel runner with live event streaming

**File:** `orchestration/runner.py` (primarily `stream_war_room`, lines ~80–146)

**Why this is the hard part (not a simple ThreadPoolExecutor + gather)**

Today `stream_war_room` is a **single-threaded generator**. It runs each specialist’s `agent.run(...)` generator to exhaustion, yielding `WarRoomEvent`s inline as they are produced, before moving to the next specialist. That is exactly why the Streamlit UI can show live per-agent progress so simply.

A naive “launch all specialists in a ThreadPoolExecutor, then gather results, then run the Manager” would:

- Lose live per-agent event streaming (the UI would only update after each specialist fully finishes, or worse, only after all of them finish).
- Break the existing generator contract that the UI depends on.

**Required design (the novel engineering)**

Preserve the generator interface while running specialists concurrently:

```
1. Create session; emit agent_start for every specialist immediately.
2. Start one worker thread (or asyncio task) per specialist.
3. Each worker:
     - runs its own agent.run(...) generator
     - pushes every AgentEvent (and agent_done) into a thread-safe queue
       tagged with agent_id
4. The main generator drains the queue (with a short timeout / sentinel)
   and yields WarRoomEvent(s) as they arrive — preserving live streaming.
5. When all workers have finished (or timed out), run the Manager once
   on the full set of structured findings and yield report / done.
```

Key implementation details to get right:

- **Thread-safe queue** (e.g. `queue.Queue`) + sentinel objects so the generator knows when a worker is finished.
- **Shared MemoryStore must be thread-safe** for concurrent `save_finding` calls (SQLite with a lock, or serialize writes through a single writer thread). This is easy to under-estimate.
- **Partial failure**: one worker raising must not kill the others; the generator still drains the remaining queue and the Manager still runs, noting failed agents in the status table.
- **Ordering of yielded events** will be non-deterministic (whichever specialist produces an event first). The UI already handles out-of-order agent events via the status snapshot; keep that invariant.
- **Sequential mode stays a pure generator path** (no threads) so the teaching toggle remains trivial and debuggable.

**Effort: L**  
This is the one item in the plan that is actual novel concurrent engineering rather than extraction/refactoring. Budget time for queue design, thread-safety of the memory store, and careful testing of interleaving + partial failure. Everything else in Phase 2 is comparatively straightforward once this works.

**UI / event requirements (unchanged, but now achievable)**
- Emit `agent_start` for every specialist at roughly the same time.
- Emit per-agent events **as they arrive** from the shared queue.
- Emit `agent_done` as each specialist finishes.
- Status table shows multiple agents in “running” state simultaneously.

**Teaching toggle**
- Sidebar (or config) option: `Execution mode = Parallel | Sequential`.
- Sequential remains the pure single-threaded generator (current behaviour).
- Document the trade-off in the report and README:

  > Specialists run concurrently for speed. They do not see each other’s findings during execution; the Manager is the only component that sees the complete picture. Switch to Sequential mode to watch one agent at a time.

**Partial failure**
- One specialist error must not cancel the others.
- Manager still runs and notes which agents failed in the status table.

### 2.2 Surface ordering / isolation decision
- In the generated report, include a short “How the team ran” section that states whether the run was parallel or sequential and what that means for finding isolation.

---

## Phase 3 — Streamlit Cloud readiness & observability

### 3.1 Pluggable memory store
- Keep `MemoryStore` interface.
- Default implementation: SQLite (local / dev).
- Add a “session-only” (in-memory) implementation for Streamlit Cloud so the app does not pretend that findings survive restarts.
- Later: optional Postgres implementation (Docker Compose).

### 3.2 Secrets & defaults for Cloud
- Read `OPENAI_API_KEY` (and optional `GITHUB_TOKEN`) from `st.secrets` when running on Streamlit Cloud.
- Default Lite mode = on for the public deployment.
- Surface the live `UsageMeter` prominently so users see token cost.

### 3.3 Basic observability
- Start simple: every `AgentEvent` (and `WarRoomEvent`) can be serialized to a JSON log line or collected into a downloadable run log.
- Add a “Download run log” button in both Single Agent and War Room UIs.
- OpenTelemetry can be a later optional extra; structured logs are enough for teaching.

---

## Phase 4 — Medium-term enhancements (optional, still valuable)

| Item | Priority | Notes |
|------|----------|-------|
| Docker Compose + optional Postgres | Medium | Makes the “real system” story concrete. Keep SQLite as default. |
| Provider selector in UI | Low | Once `LLMProvider` abstraction exists, exposing Anthropic / others is straightforward. |
| Richer comparison tab | Low | Single-agent answer vs full War Room report side-by-side with token cost. |
| Separate “chat with repo docs” RAG track | Low | Only if a RAG demo is explicitly desired; **not** part of the War Room security path. |

---

## Implementation order (recommended)

| Order | Task | Phase | Effort | Impact |
|-------|------|-------|--------|--------|
| 1 | Fix clone URL + Limitations block | 0 | XS | Honesty & polish |
| 2 | Update Technical Decisions Log | 0 | XS | Documentation |
| 3 | Extract magic numbers → `Limits` | 1 | S | Clarity, tunability |
| 4 | `LLMProvider` abstraction + injectable client | 1 | M | Educational extension point |
| 5 | Structured findings only | 1 | S | Robustness |
| 6 | Tavily CVE tool for Security Specialist | 1 | S | Upgrades weakest analysis area; high demo value |
| 7 | Parallel specialist runner + live event streaming + sequential toggle | 2 | L | Latency, realism, teaching; **novel concurrent engineering** |
| 8 | Expand Learning Path exercises | 1 | S | Turns repo into a real tutorial |
| 9 | Pluggable memory + Cloud defaults | 3 | M | Streamlit Cloud viability |
| 10 | Downloadable run log | 3 | S | Observability for learners |
| 11 | Docker Compose / Postgres | 4 | L | Portfolio depth |

---

## Testing expectations

- All existing 169 tests must continue to pass.
- New tests required for:
  - `Limits` values actually applied (tool result truncation, step budgets, etc.).
  - `LLMProvider` injection (mock provider).
  - `search_cve` tool (happy path, missing API key, rate/size limits, structured output shape).
  - Parallel runner:
    - specialists finish independently;
    - events are yielded as they arrive (not only after each agent completes);
    - Manager sees all findings;
    - partial failure (one specialist errors, others succeed, Manager still runs);
    - MemoryStore remains consistent under concurrent writes.
  - Limitations block always present in assembled reports.
- Keep Lite-mode tests fast; full-mode tests can stay marked or run only in CI.

---

## Success criteria

- A learner can follow the updated Learning Path and understand every major design decision.
- Parallel War Room runs are noticeably faster than sequential on the same repository.
- Switching between Parallel / Sequential is one click and the difference is explained in the report.
- Streamlit Cloud deployment works with secrets, defaults to Lite mode, and does not claim persistent memory.
- The Technical Decisions Log accurately reflects the current architecture.
- Cost of a typical single-agent + parallel War Room run remains well under $1 (usually < $0.30).

---

## Out of scope for this patch

- Full multi-user authentication.
- Private repository support.
- Replacing the custom loop with LangGraph / CrewAI.
- Production-grade rate limiting or billing protection beyond a simple demo guard.
- Real-time collaborative editing of findings.

---

*This plan is intentionally incremental. Each phase leaves the project in a working, teachable state.*
