"""
GitHub Agent Studio — Main Streamlit Application

A thin UI wiring layer that orchestrates investigation modes.
All business logic lives in isolated modules (agents, tools, memory, orchestration).

This app stays minimal and focuses solely on:
- Displaying the UI shell
- Routing user input to appropriate orchestration functions
- Rendering results

Two investigation modes:
1. Single Agent Mode: Transparent single-agent loop for learning
2. Multi-Agent War Room: Specialized agents coordinated by a manager
"""

import json

import streamlit as st

from agents.single import SingleAgent
from memory import MemoryStore
from prompts.single_agent import QUICK_PROMPTS
from tools import parse_repo_ref
from ui.components import memory_panel, render_events


@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore()


def render_header():
    st.set_page_config(page_title="GitHub Agent Studio", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 GitHub Agent Studio")


def load_config():
    """Settings need OPENAI_API_KEY; show a friendly message instead of a traceback when it is missing."""
    try:
        from config import config

        return config
    except Exception:
        st.error("**OPENAI_API_KEY is not set.** Copy `.env.example` to `.env`, add your key, and restart the app.")
        st.stop()


def load_chat(store: MemoryStore, repo_id: str) -> dict:
    """Chat state for a repository; resumes the most recent single-agent session from memory if there is one."""
    chat = st.session_state.get("chat")
    if chat and chat["repo_id"] == repo_id:
        return chat
    sessions = store.list_sessions(repo_id, mode="single_agent", limit=1)
    session_id = sessions[0].session_id if sessions else None
    messages = []
    for m in store.get_conversation_history(session_id) if session_id else []:
        tools_used = [json.loads(t)["name"] for t in m.tool_calls or []]
        messages.append({"role": m.role, "content": m.content, "events": [], "tools": tools_used})
    st.session_state.chat = {"repo_id": repo_id, "session_id": session_id, "messages": messages}
    return st.session_state.chat


def render_sidebar(config, store: MemoryStore, repo_id: str | None):
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio(
        "Investigation Mode",
        options=["single_agent", "multi_agent"],
        format_func=lambda x: "Single Agent (Agent 101)" if x == "single_agent" else "Multi-Agent War Room",
    )
    st.sidebar.header("📦 Repository")
    st.sidebar.text_input("GitHub repository (owner/repo or URL)", placeholder="langchain-ai/langchain", key="repo_text")

    st.sidebar.header("⚡ Quick actions")
    for key, label in (("analyze", "🔍 Analyze this repository"), ("teach", "🎓 Teach me this repo"), ("continue", "▶️ Continue where we stopped")):
        if st.sidebar.button(label, use_container_width=True, disabled=repo_id is None or mode != "single_agent"):
            st.session_state.pending_prompt = QUICK_PROMPTS[key]

    st.sidebar.header("💾 Memory")
    if repo_id:
        if st.sidebar.button("🆕 New session", use_container_width=True, help="Start a fresh conversation; long-term memory is kept."):
            chat = st.session_state.get("chat")
            if chat and chat["session_id"]:
                store.complete_session(chat["session_id"])
            st.session_state.chat = {"repo_id": repo_id, "session_id": None, "messages": []}
        st.sidebar.download_button(
            "⬇️ Export memory (JSON)", json.dumps(store.export_repository_memory(repo_id), indent=2),
            file_name=f"{repo_id.replace('/', '_')}_memory.json", mime="application/json", use_container_width=True,
        )
        if st.sidebar.button("🗑️ Clear repository memory", use_container_width=True):
            store.clear_repository_memory(repo_id)
            st.session_state.chat = {"repo_id": repo_id, "session_id": None, "messages": []}
            st.sidebar.success("Memory cleared for this repository.")

    st.sidebar.info(
        f"**Model:** {config.openai_model}  \n"
        f"**GitHub token:** {'✅ configured' if config.github_token else '❌ not set (60 requests/hour)'}"
    )
    return mode


def render_chat_history(chat: dict):
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            if message.get("events"):
                with st.expander(f"🔎 Investigation steps ({len(message['events'])})"):
                    render_events(message["events"])
            elif message.get("tools"):
                st.caption(f"🔧 Used {len(message['tools'])} tool call(s): {', '.join(message['tools'])}")
            st.markdown(message["content"])


def run_turn(store: MemoryStore, config, chat: dict, owner: str, repo: str, prompt: str):
    """Run one user turn, streaming every agent step into the page as it happens."""
    agent = SingleAgent(store, max_output_tokens=config.max_output_tokens)
    if chat["session_id"] is None:
        chat["session_id"] = agent.start_session(owner, repo)
    chat["messages"].append({"role": "user", "content": prompt, "events": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Investigating…", expanded=True)
        answer_slot = st.empty()
        events = []
        for event in agent.run(owner, repo, prompt, chat["session_id"]):
            events.append(event)
            with status:
                render_events([event], live=True)
        last = events[-1]
        steps = max((e.step for e in events), default=0)
        status.update(label="Something went wrong" if last.kind == "error" else f"Done: {steps} tool round(s)",
                      state="error" if last.kind == "error" else "complete", expanded=False)
        answer = last.content if last.kind == "final" else ""
        answer_slot.markdown(answer)
    chat["messages"].append({"role": "assistant", "content": answer, "events": [e for e in events if e.kind != "final"]})


def main():
    render_header()
    config = load_config()
    store = get_store()

    # The sidebar text box has key "repo_text", so its value is already known before the sidebar is drawn.
    repo_text = st.session_state.get("repo_text", "").strip()
    try:
        ref = parse_repo_ref(repo_text) if repo_text else None
    except ValueError:
        ref = None
    repo_id = MemoryStore.make_repo_id(ref.owner, ref.repo) if ref else None

    mode = render_sidebar(config, store, repo_id)

    if mode == "multi_agent":
        st.info("The Multi-Agent War Room arrives in Phases 4–5. Switch to Single Agent mode to investigate now.")
        return
    if not repo_text:
        st.info("Enter a public repository in the sidebar (for example `pallets/click`) to begin.")
        return
    if ref is None:
        st.error(f"“{repo_text}” is not a valid repository. Use `owner/repo` or a github.com URL.")
        return

    chat = load_chat(store, repo_id)
    st.caption(f"Investigating **{repo_id}**" + (" · resumed previous session" if chat["messages"] else ""))
    chat_tab, memory_tab = st.tabs(["💬 Investigation", "🧠 Memory"])

    prompt = st.chat_input("Ask about this repository…") or st.session_state.pop("pending_prompt", None)
    with chat_tab:
        render_chat_history(chat)
        if prompt:
            run_turn(store, config, chat, ref.owner, ref.repo, prompt)
    with memory_tab:  # rendered after the turn so it reflects anything just saved
        memory_panel(store, repo_id)


if __name__ == "__main__":
    main()
