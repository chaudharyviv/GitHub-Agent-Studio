"""
GitHub Agent Studio — Main Streamlit Application

A thin UI wiring layer that orchestrates investigation modes.
All business logic lives in isolated modules (agents, tools, memory, orchestration);
all drawing lives in ``ui/``. This file only:
- sets up the page, configuration and memory store
- draws the sidebar and the repository card
- hands the chosen repository to the screen for the chosen mode

Two investigation modes:
1. Single Agent Mode: Transparent single-agent loop for learning
2. Multi-Agent War Room: Specialized agents coordinated by a manager
"""

import streamlit as st

from memory import MemoryStore
from tools import get_repository, is_error
from tools.schemas import RepositoryInput
from ui.components import repo_header, safe_render
from ui.sidebar import render_sidebar
from ui.single_agent import render_single_agent
from ui.styles import apply_theme
from ui.war_room import render_war_room


@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore()


def load_config():
    """Settings need OPENAI_API_KEY; show a friendly message instead of a traceback when it is missing."""
    try:
        from config import config

        return config
    except Exception:
        st.error("**OPENAI_API_KEY is not set.** Copy `.env.example` to `.env`, add your key, and restart the app.")
        st.stop()


def show_repository(owner: str, repo: str) -> bool:
    """Look the repository up and draw its card. Returns False if it cannot be investigated."""
    with st.spinner("Checking repository…"):
        info = get_repository(RepositoryInput(owner=owner, repo=repo))
    if not is_error(info):
        repo_header(info)
        return True
    if info.kind == "not_found":
        st.error(f"**{owner}/{repo}** was not found. Check the spelling, and note that private repositories are not supported.")
        return False
    if info.kind == "rate_limited":
        st.warning(f"{info.message} You can still continue with data already cached, but new lookups will fail.")
    else:
        st.warning(f"Could not load the repository card: {info.message}")
    return True


def run_app():
    config = load_config()
    store = get_store()
    side = render_sidebar(config, store)

    if not side.repo_text:
        st.info("Enter a public repository in the sidebar (for example `pallets/click`) to begin.")
        return
    if side.repo_id is None:
        st.error(f"“{side.repo_text}” is not a valid repository. Use `owner/repo` or a github.com URL.")
        return
    if not show_repository(side.owner, side.repo):
        return

    screen = render_war_room if side.mode == "multi_agent" else render_single_agent
    screen(store, config, side.owner, side.repo, side.repo_id)


def main():
    st.set_page_config(page_title="GitHub Agent Studio", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    st.title("🤖 GitHub Agent Studio")
    safe_render(run_app)


if __name__ == "__main__":
    main()
