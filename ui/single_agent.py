"""
The Single Agent screen: a chat with a transparent agent that remembers what it learns.

Every step the agent takes is streamed into the page as it happens, and replayed
from the chat history afterwards.
"""

import json

import streamlit as st

from agents.single import SingleAgent
from memory import MemoryStore
from ui.components import memory_panel, render_events, tool_timeline


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


def render_chat_history(chat: dict):
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            if message.get("events"):
                with st.expander(f"🔎 Investigation steps ({len(message['events'])})"):
                    tool_timeline(message["events"])
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
        st.caption(f"💲 {agent.usage.summary()}")
    chat["messages"].append({"role": "assistant", "content": answer, "events": [e for e in events if e.kind != "final"]})


def render_single_agent(store: MemoryStore, config, owner: str, repo: str, repo_id: str):
    chat = load_chat(store, repo_id)
    st.caption(f"Investigating **{repo_id}**" + (" · resumed previous session" if chat["messages"] else ""))
    chat_tab, memory_tab = st.tabs(["💬 Investigation", "🧠 Memory"])

    prompt = st.chat_input("Ask about this repository…") or st.session_state.pop("pending_prompt", None)
    with chat_tab:
        render_chat_history(chat)
        if prompt:
            run_turn(store, config, chat, owner, repo, prompt)
    with memory_tab:  # rendered after the turn so it reflects anything just saved
        memory_panel(store, repo_id)
