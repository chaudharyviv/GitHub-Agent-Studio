# GitHub Agent Studio

An educational AI agent project that investigates any public GitHub repository using OpenAI's `gpt-4o-mini` model.

**Status:** Feature-complete for v1 — both investigation modes work end-to-end, with a persistent SQLite memory layer and 169 passing tests. See `spec.md` and `plan.md` for the original design and roadmap this was built from.

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
A team of specialized agents coordinated by a manager:
- **Architecture Specialist** — Repository structure and design patterns
- **Security Specialist** — Dependency risks and security posture
- **Code Quality Specialist** — Code organization and test coverage
- **Project Health Specialist** — Activity, contributors, release velocity
- **Manager Agent** — Synthesizes findings into a final health report

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (get it at [platform.openai.com](https://platform.openai.com))

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/your-org/github-agent-studio.git
   cd github-agent-studio
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
│   ├── loop.py                 # Shared OpenAI tool-calling loop
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
├── tools/                    # GitHub API tools
│   ├── github.py             # Tool implementations
│   ├── client.py              # HTTP client + rate limiting
│   ├── cache.py                # Response caching
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

| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python 3.11+ | Ecosystem + Streamlit + OpenAI library |
| UI | Streamlit | Fast path to polished demo |
| LLM | OpenAI gpt-4o-mini | Best cost/speed/quality for tool-calling loops |
| Database | SQLite | Simple, local, sufficient for v1 |
| Agent Framework | None (custom loops) | Maximum transparency for learning |
| GitHub Client | httpx + Pydantic | Lightweight, full control |

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
```

`MAX_OUTPUT_TOKENS` and `LITE_MODE` are just the defaults the sidebar starts from — anyone running the app can override either one per session without touching `.env`.

## Learning Path

New to AI agents? Start here:

1. **Understand the structure** → Read `spec.md` and `plan.md`
2. **Explore the code** → Browse the module organization in `agents/`, `tools/`, `memory/`
3. **Study the single-agent loop** → `agents/single.py` and the shared tool-calling loop in `agents/loop.py`
4. **Study the multi-agent team** → `agents/multi/specialist.py` (shared base class), then the Manager in `agents/multi/manager.py` and orchestration in `orchestration/runner.py`
5. **Compare modes** → Run the same repository through both modes and compare via the War Room's "Single vs Team" tab

## Non-Goals (v1)

- Private repository support
- Real vulnerability scanning (static heuristics only)
- Full static analysis engines
- Full multi-user authentication (there's a lightweight opt-in name field to keep concurrent people's chats separate on a shared instance — see "Using the app" above — but no accounts, passwords, or verified identity)
- Production cloud deployment
- Support for non-OpenAI LLM providers
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
