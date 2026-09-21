# GitHub Agent Studio

An educational AI agent project that investigates any public GitHub repository using OpenAI's `gpt-4o-mini` model.

**Status:** Phase 0 (Project Skeleton) — See `spec.md` and `plan.md` for full roadmap.

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

## Current Status

**Phase 0 — Project Skeleton** ✅ Complete
- ✅ Directory structure and module organization
- ✅ Configuration management (pydantic-settings)
- ✅ Type schemas (Pydantic models)
- ✅ Agent base classes and stubs
- ✅ Minimal Streamlit UI shell
- ✅ Database schema design

**Upcoming Phases**
- Phase 1: GitHub Tools Layer (fetch data from GitHub API)
- Phase 2: Memory Layer (SQLite persistence)
- Phase 3: Single Agent Mode (full investigation loop)
- Phase 4: Multi-Agent Specialists (four specialized agents)
- Phase 5: Manager + Orchestration (synthesis and reports)
- Phase 6: UI Polish & Integration (complete user experience)
- Phase 7: Documentation & Demo Polish

See `plan.md` for detailed timeline and `spec.md` for complete requirements.

## Architecture

```
github-agent-studio/
├── app.py                    # Thin Streamlit UI wiring
├── config.py                 # Environment configuration
├── agents/                   # Agent implementations
│   ├── base.py              # Base agent class
│   ├── single.py            # Single agent loop
│   └── multi/               # Specialist agents
│       ├── manager.py
│       ├── architecture.py
│       ├── security.py
│       ├── quality.py
│       └── health.py
├── tools/                    # GitHub API tools
│   ├── github.py            # Tool implementations
│   └── schemas.py           # Pydantic input/output models
├── memory/                   # SQLite persistence
│   ├── store.py             # Database layer
│   └── schemas.py           # Memory data models
├── orchestration/           # Agent coordination
│   └── runner.py            # Single and multi-agent runners
├── ui/                      # Streamlit components
│   ├── components.py        # Reusable UI widgets
│   └── styles.py            # Theming and styling
├── prompts/                 # LLM system prompts
│   ├── single_agent.py
│   └── multi_agent.py
├── spec.md                  # Full feature specification
├── plan.md                  # Implementation roadmap
└── requirements.txt         # Python dependencies
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

All configuration comes from environment variables or `.env`:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_MODEL=gpt-4o-mini    # Default: gpt-4o-mini (locked for v1)
GITHUB_TOKEN=ghp_...         # Recommended for higher rate limits
```

## Learning Path

New to AI agents? Start here:

1. **Understand the structure** → Read `spec.md` and `plan.md`
2. **Explore the code** → Browse the module organization in `agents/`, `tools/`, `memory/`
3. **Study the skeleton** → See how agent types are defined in `agents/base.py` and `agents/single.py`
4. **Follow Phase 3** → Once Phase 3 is implemented, study the single-agent loop code
5. **Compare modes** → Once Phase 5 is complete, compare single vs. multi-agent approaches

## Non-Goals (v1)

- Private repository support
- Real vulnerability scanning (static heuristics only)
- Full static analysis engines
- Multi-user authentication
- Production cloud deployment
- Support for non-OpenAI LLM providers
- LangGraph, CrewAI, or similar frameworks in core path

## Contributing

This is an educational portfolio project. Contributions are welcome!

- Follow the existing code structure and type patterns
- Keep `app.py` thin — all logic lives in modules
- Use Pydantic models for all data transfer
- Document all agent responsibilities clearly
- Test locally before submitting changes

## License

MIT (see LICENSE file)

## References

- [GitHub REST API](https://docs.github.com/en/rest)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Pydantic Documentation](https://docs.pydantic.dev)

---

**Note:** This project is an educational demonstration of AI agent patterns. It is not intended for production use without significant additional work on error handling, rate limiting, and security.
