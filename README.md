# GitHub Agent Studio

An educational AI agent project that investigates any public GitHub repository using OpenAI's `gpt-4o-mini` model.

**Status:** Feature-complete for v1 — both investigation modes work end-to-end, with a pluggable (SQLite or in-memory) memory layer and 197 passing tests. See `spec.md` and `plan.md` for the original design and roadmap this was built from.

## Overview

GitHub Agent Studio demonstrates modern AI agent patterns with two investigation modes inside a single Streamlit application:

### 🤖 Single Agent Mode (Agent 101)
A transparent, educational single agent that:
- Investigates any public GitHub repository
- Decides which tools to call and why
- Maintains memory across conversations
- Shows every decision and tool call
- Perfect for learning how agents work

### 🏛️ Multi-Agent War Room Mode
A team of specialized agents coordinated by a manager, run sequentially (default) or in parallel (one click to switch):
- **Architecture Specialist** — Repository structure and design patterns
- **Security Specialist** — Dependency risks and security posture, with optional live CVE lookup (Tavily → NVD / GitHub Advisories)
- **Code Quality Specialist** — Code organization and test coverage
- **Project Health Specialist** — Activity, contributors, release velocity
- **Manager Agent** — Synthesizes findings into a final health report

Sequential lets each specialist see what earlier ones already found, and is easier to watch one step at a time. Parallel runs all four at once for speed, at the cost of that cross-specialist visibility during the run — the report always states which mode produced it.

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (get it at [platform.openai.com](https://platform.openai.com))

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/chaudharyviv/GitHub-Agent-Studio.git
   cd GitHub-Agent-Studio
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file from `.env.example`
   ```bash
   cp .env.example .env
   ```
   
5. Edit `.env` and add your OpenAI API key
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

6. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

7. Open your browser to `http://localhost:8501`

## Using the app

The sidebar controls apply to both modes:

- **Investigation mode** — Single Agent or Multi-Agent War Room.
- **🪶 Lite mode** — shrinks tool results and tool-calling rounds for cheap test runs (see `agents/limits.py`); off by default for thorough investigations.
- **Max output tokens** — cap on how much the model may write per call (256–16,384); higher allows longer answers and reports at higher cost.
- **👤 Your name** — optional. Leave blank to share one continuous history per repository (the default, right for a single local user). If more than one person uses the same running instance, each person can set a name so their chat doesn't get mixed up with someone else's. Findings and War Room reports are always shared team knowledge regardless of this setting — only the live *conversation* is scoped by name.

Type a repository as `owner/repo` or a full GitHub URL, then either chat with the single agent or run the War Room team.

## Architecture

```
github-agent-studio/
├── app.py                    # Thin Streamlit UI wiring
├── config.py                 # Environment configuration
├── agents/                   # Agent implementations
│   ├── base.py               # Base agent class
│   ├── single.py              # Single agent loop
│   ├── toolbox.py             # Tool dispatch + memory side effects
│   ├── loop.py                 # Shared tool-calling loop (talks to an LLMProvider, never a raw SDK client)
│   ├── llm.py                    # LLMProvider abstraction + OpenAIProvider (the multi-provider extension point)
│   ├── limits.py                # Lite/Full mode size limits
│   ├── usage.py                  # Token/cost accounting
│   └── multi/                   # Specialist agents
│       ├── manager.py
│       ├── specialist.py         # Shared specialist base class
│       ├── architecture.py
│       ├── security.py
│       ├── quality.py
│       ├── health.py
│       └── report.py             # Health report assembly
├── tools/                    # GitHub API tools + CVE lookup
│   ├── github.py             # Tool implementations
│   ├── client.py              # HTTP client + rate limiting
│   ├── cache.py                # Response caching
│   ├── cve.py                   # Tavily-backed search_cve (Security Specialist only)
│   └── schemas.py             # Pydantic input/output models
├── memory/                   # SQLite persistence
│   ├── store.py               # Database layer
│   └── schemas.py              # Memory data models
├── orchestration/            # Agent coordination
│   └── runner.py              # Single and multi-agent runners
├── ui/                       # Streamlit components
│   ├── sidebar.py             # Mode, repository, memory controls
│   ├── single_agent.py         # Single Agent screen
│   ├── war_room.py              # War Room screen
│   ├── components.py            # Reusable UI widgets
│   └── styles.py                # Theming and styling
├── prompts/                  # LLM system prompts
│   ├── single_agent.py
│   └── multi_agent.py
├── tests/                    # pytest suite (169 tests)
├── spec.md                   # Full feature specification
├── plan.md                   # Implementation roadmap
└── requirements.txt          # Python dependencies
```

## Key Technical Decisions

See `spec.md` (§7 Technical Constraints & Decisions) for the full table with alternatives considered. Summary:

| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python 3.11+ | Ecosystem + Streamlit + OpenAI library |
| UI | Streamlit | Fast path to polished demo |
| Agent Framework | None (custom loops) | Maximum transparency for learning |
| Multi-agent execution | Parallel or sequential specialists (toggle), Manager merge | Sequential (default) is simple to follow/debug; parallel trades that for speed |
| Database | SQLite (pluggable) | Simple, local, sufficient for v1; easy to swap later |
| GitHub Client | httpx + Pydantic | Lightweight, full control |
| LLM | OpenAI gpt-4o-mini, via a thin `LLMProvider` abstraction (`agents/llm.py`) | Best cost/speed/quality for tool-calling loops today; the abstraction keeps a second provider a one-file addition |
| Findings format | Structured (Pydantic / tool args only) | Reliable, machine-readable |

## Configuration

All configuration comes from environment variables or `.env` (see `.env.example`):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_MODEL=gpt-4o-mini     # Default: gpt-4o-mini (locked for v1)
MAX_OUTPUT_TOKENS=2048        # Default per-call output cap; also adjustable live in the sidebar (256-16384)
LITE_MODE=0                    # Default cheap-test-run toggle; also adjustable live in the sidebar
GITHUB_TOKEN=ghp_...             # Recommended for higher rate limits
TAVILY_API_KEY=tvly-...           # Optional; enables the Security Specialist's live CVE lookup
MEMORY_BACKEND=sqlite              # "sqlite" (persists) or "memory" (session-only; see Streamlit Cloud below)
```

`MAX_OUTPUT_TOKENS` and `LITE_MODE` are just the defaults the sidebar starts from — anyone running the app can override either one per session without touching `.env`.

## Deploying to Streamlit Cloud

The app runs there as-is, with two things worth setting explicitly in the deployment's **Secrets** (Streamlit Cloud reads `st.secrets`, not a `.env` file — `app.py` copies whatever is in `st.secrets` into the environment at startup, so every variable above works the same way there):

```toml
# .streamlit/secrets.toml (or the Secrets box in Streamlit Cloud's app settings)
OPENAI_API_KEY = "sk-..."
MEMORY_BACKEND = "memory"   # Cloud's filesystem is not durable either; don't let SQLite claim otherwise
LITE_MODE = "1"             # cheaper default for a public, unauthenticated deployment
```

With `MEMORY_BACKEND=memory`, findings, reports and conversations live only as long as the app process does — the sidebar says so under **Memory**. The sidebar also shows a running **Session cost** (all LLM spend so far this browser session), so a public deployment never surprises anyone.

## Learning Path

New to AI agents? Start here, then work through the exercises below — each names the exact files to open and a concrete thing to try:

1. **Understand the structure** → Read `spec.md` and `plan.md`
2. **Explore the code** → Browse the module organization in `agents/`, `tools/`, `memory/`
3. **Study the single-agent loop** → `agents/single.py` and the shared tool-calling loop in `agents/loop.py`
4. **Study the multi-agent team** → `agents/multi/specialist.py` (shared base class), then the Manager in `agents/multi/manager.py` and orchestration in `orchestration/runner.py`
5. **Compare modes** → Run the same repository through both modes and compare via the War Room's "Single vs Team" tab

### Exercises

1. **Watch a single tool-calling loop.**
   Read `agents/loop.py` (`run_tool_loop`), then `agents/single.py`. Run Single Agent on `pallets/click` and ask "What does this repo do?" — watch the reasoning → tool_call → tool_result events stream live before the final answer.

2. **Memory side-effects.**
   Read `agents/toolbox.py` (`_save_finding`, `_after_call`). Ask a question that produces a finding (e.g. "Does this repo have a test suite?"), then start a new conversation on the same repository and ask "What have you found so far?" — the earlier finding comes back from SQLite, not from chat history.

3. **Compare Single vs War Room.**
   Run the same repository through both modes, then open the War Room's "Single vs Team" tab: the single agent's actual last answer sits next to the Manager's narrative, each with its real cost (only shown for turns run in this browser session — an older, reloaded answer honestly shows no cost rather than a stale or fabricated one), and a findings-by-category comparison below that.

4. **Limits & cost control.**
   Read `agents/limits.py` (`Limits`, `NORMAL`, `LITE`). Toggle Lite mode in the sidebar, change max-output-tokens, re-run the same repository, and compare token usage (shown under each run) and report length.

5. **The `LLMProvider` extension point.**
   Read `agents/llm.py` (`LLMProvider`, `OpenAIProvider`) and where `resolve_llm` is called from `agents/single.py`, `agents/multi/specialist.py`, `agents/multi/manager.py`, and `orchestration/runner.py`. Notice that `agents/loop.py`'s `run_tool_loop` never imports `openai` — it only calls `provider.chat(...)`. `tests/test_agent.py::test_explicit_provider_is_used_and_client_model_are_ignored` shows a second provider plugged in with no `agents/loop.py` changes.

6. **Live CVE lookup, and its limits.**
   Read `tools/cve.py` (`search_cve`) and the Security Specialist's prompt in `prompts/multi_agent.py` (`get_security_prompt`). Without `TAVILY_API_KEY` set, run the War Room and open the Security specialist's tool calls — `search_cve` returns a clear "unavailable" error instead of a guess. With a key set, look for a returned `cve_id` in the report's evidence appendix and note the report's Limitations section: it is a search lead, not a verified database match.

7. **Parallel vs sequential.**
   Read `orchestration/runner.py` (`_run_sequential`, `_run_parallel`) — same specialists, same generator interface, but `_run_parallel` puts each specialist on its own thread and merges their events through a `queue.Queue`. Run the War Room once with each Execution mode toggle and compare: wall-clock time, whether the status dashboard shows several specialists "running" at once, and the report's "How the team ran" section. Then read `tests/test_war_room.py`'s `RoutedFakeOpenAI` to see how a scripted LLM is made safe for concurrent specialist threads (it routes each request by a marker in the system prompt instead of assuming a fixed call order).

8. **Pluggable memory, and honest persistence.**
   Read `app.py` (`get_store`, `_load_secrets_into_env`) and `config.py`'s `memory_backend` field. Run locally with `MEMORY_BACKEND=memory` in `.env`: the sidebar's **Memory** line switches to "in-memory", and restarting the app (not just reloading the browser) loses everything — the point being that the UI never claims persistence the current backend can't actually offer. Then look at the **Session cost** metric in the sidebar and `ui/components.py::get_session_usage` to see how it stays accurate even though the sidebar is drawn *before* the screen that spends the money (`ui/sidebar.py`'s `cost_slot` placeholder, refreshed by `app.py` after the screen runs).

## Non-Goals (v1)

- Private repository support
- Real vulnerability scanning or SCA (static heuristics + optional live CVE search leads, not a verified database match)
- Full static analysis engines
- Full multi-user authentication (there's a lightweight opt-in name field to keep concurrent people's chats separate on a shared instance — see "Using the app" above — but no accounts, passwords, or verified identity)
- Production-grade rate limiting or billing protection (the session cost display is informational, not enforced)
- Support for non-OpenAI LLM providers (the `LLMProvider` abstraction exists; no second provider ships)
- LangGraph, CrewAI, or similar frameworks in core path

## Contributing

This is an educational portfolio project. Contributions are welcome!

- Follow the existing code structure and type patterns
- Keep `app.py` thin — all logic lives in modules
- Use Pydantic models for all data transfer
- Document all agent responsibilities clearly
- Test locally before submitting changes (`pytest`)

## License

MIT (see LICENSE file)

## References

- [GitHub REST API](https://docs.github.com/en/rest)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Pydantic Documentation](https://docs.pydantic.dev)

---

**Note:** This project is an educational demonstration of AI agent patterns. It is not intended for production use without significant additional work on error handling, rate limiting, and security — in particular, real authentication if it's ever deployed for more than one person.
