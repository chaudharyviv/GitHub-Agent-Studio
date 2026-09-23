"""
The sidebar: mode, repository, quick actions, memory controls and configuration.

The repository text lives in session state under its own key, so switching mode never
loses it; both screens read the repository from here.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from memory import MemoryStore
from prompts.single_agent import QUICK_PROMPTS
from ui.components import github_rate_limit_caption, identity_input, mode_selector, repository_input, runtime_settings, session_cost_metric


@dataclass
class SidebarState:
    mode: str  # "single_agent" or "multi_agent"
    repo_text: str  # what the user typed
    owner: Optional[str]  # None when the text is empty or not a valid repository
    repo: Optional[str]
    repo_id: Optional[str]
    lite_mode: bool
    max_output_tokens: int
    identity: Optional[str]  # optional name; keeps this person's chat separate on a shared instance
    cost_slot: Any  # st.empty() placeholder in the sidebar; the caller refreshes it after the screen
    # runs, since the sidebar is drawn before the turn that updates the session cost happens


def reset_screens(keep_new_chat_for: Optional[str] = None):
    """Forget the on-screen chat and War Room state (memory itself is untouched)."""
    st.session_state.pop("war_room", None)
    if keep_new_chat_for:  # a fresh, empty conversation that does not resume the last session
        st.session_state.chat = {"repo_id": keep_new_chat_for, "session_id": None, "messages": []}
    else:
        st.session_state.pop("chat", None)


def _memory_controls(store: MemoryStore, repo_id: str, mode: str):
    st.header("💾 Memory")
    if mode == "single_agent" and st.button("🆕 New session", width="stretch", help="Start a fresh conversation; long-term memory is kept."):
        chat = st.session_state.get("chat")
        if chat and chat["session_id"]:
            store.complete_session(chat["session_id"])
        reset_screens(keep_new_chat_for=repo_id)

    st.download_button(
        "⬇️ Export memory (JSON)", json.dumps(store.export_repository_memory(repo_id), indent=2),
        file_name=f"{repo_id.replace('/', '_')}_memory.json", mime="application/json", width="stretch",
    )

    if st.button("🗑️ Clear repository memory", width="stretch"):
        st.session_state.confirm_clear = repo_id
    if st.session_state.get("confirm_clear") == repo_id:
        context = store.get_resume_context(repo_id, max_findings=10_000)
        st.warning(
            f"Permanently delete **{len(context.findings)} findings**, {len(store.list_sessions(repo_id))} session(s), "
            f"saved reports and your notes for **{repo_id}**? This cannot be undone."
        )
        confirm, cancel = st.columns(2)
        if confirm.button("Delete", type="primary", key="confirm_clear_yes", width="stretch"):
            store.clear_repository_memory(repo_id)
            reset_screens()
            st.session_state.confirm_clear = None
            st.toast(f"Memory cleared for {repo_id}", icon="🗑️")
            st.rerun()
        if cancel.button("Cancel", key="confirm_clear_no", width="stretch"):
            st.session_state.confirm_clear = None
            st.rerun()


def render_sidebar(config, store: MemoryStore) -> SidebarState:
    with st.sidebar:
        st.header("⚙️ Mode")
        mode = mode_selector()
        lite_mode, max_output_tokens = runtime_settings(config)
        identity = identity_input()
        st.header("📦 Repository")
        owner, repo = repository_input()
        repo_id = MemoryStore.make_repo_id(owner, repo) if owner and repo else None

        if repo_id and mode == "single_agent":
            st.header("⚡ Quick actions")
            for key, label in (("analyze", "🔍 Analyze this repository"), ("teach", "🎓 Teach me this repo"), ("continue", "▶️ Continue where we stopped")):
                if st.button(label, width="stretch"):
                    st.session_state.pending_prompt = QUICK_PROMPTS[key]
        if repo_id:
            _memory_controls(store, repo_id, mode)

        # a placeholder, not a direct call: this point in the script runs *before* the screen below it,
        # so a direct call here would always show last run's total, one turn stale; the caller fills it
        # in (twice: once now so it is never blank, again after the screen updates the running total)
        cost_slot = st.empty()
        with cost_slot:
            session_cost_metric()
        st.info(
            f"**Model:** {config.openai_model}{' · 🪶 lite mode' if lite_mode else ''}  \n"
            f"**GitHub token:** {'✅ configured' if config.github_token else '❌ not set (60 requests/hour)'}  \n"
            f"**Memory:** {'🧠 in-memory (forgotten when this app restarts)' if config.memory_backend == 'memory' else '💾 persistent (SQLite)'}"
        )
        github_rate_limit_caption()
    return SidebarState(
        mode=mode, repo_text=st.session_state.get("repo_text", "").strip(), owner=owner, repo=repo, repo_id=repo_id,
        lite_mode=lite_mode, max_output_tokens=max_output_tokens, identity=identity, cost_slot=cost_slot,
    )
