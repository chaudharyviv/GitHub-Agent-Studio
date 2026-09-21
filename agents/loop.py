"""
The tool-calling loop shared by every agent (single agent and all specialists).

    repeat:
        reply = LLM(messages, tools)
        no tool calls?  -> that is the answer, stop
        otherwise       -> run each requested tool, append the results, go again

It is a generator: each step is yielded as an ``AgentEvent`` the moment it
happens, so a UI can show the agent's work live. ``messages`` is extended in
place, which lets a caller keep the conversation going after the loop ends.
"""

import json
from typing import Any, Iterator, Optional

from agents.base import AgentEvent
from agents.toolbox import Toolbox

# Cap on tokens the model may write per call. Tool-call rounds need ~100-400 (a sentence + arguments);
# the longest answer ("Teach me this repo") runs about 1,200-1,800. 2,048 fits that with headroom
# while keeping a runaway reply cheap. gpt-4o-mini's hard limit is 16,384.
DEFAULT_MAX_OUTPUT_TOKENS = 2048
TRUNCATION_NOTE = "\n\n_(The answer hit the output token limit and was cut off. Ask me to continue.)_"


def resolve_llm(client: Any = None, model: Optional[str] = None) -> tuple[Any, str]:
    """The OpenAI client and model to use; built from OPENAI_API_KEY / OPENAI_MODEL when not injected."""
    if client is None:
        from openai import OpenAI

        from config import config  # imported lazily: it requires OPENAI_API_KEY to be set

        client, model = OpenAI(api_key=config.openai_api_key), model or config.openai_model
    return client, model or "gpt-4o-mini"


def run_tool_loop(
    client: Any,
    model: str,
    messages: list[dict],
    toolbox: Toolbox,
    *,
    max_steps: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Iterator[AgentEvent]:
    """
    Run the loop until the model answers. The last event yielded is 'final' or 'error'.

    After ``max_steps`` tool-calling rounds the model is told to answer without tools.
    The final assistant message is appended to ``messages``.
    """
    for step in range(1, max_steps + 2):
        can_use_tools = step <= max_steps
        if not can_use_tools:
            messages.append({"role": "system", "content": "Tool budget used up. Answer now using only what you have gathered, and say what you did not get to check."})
        try:
            kwargs = {"tools": toolbox.specs(), "tool_choice": "auto"} if can_use_tools else {}
            choice = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_completion_tokens=max_output_tokens, **kwargs
            ).choices[0]
            reply = choice.message
        except Exception as exc:
            yield AgentEvent(kind="error", step=step, is_error=True, content=explain_llm_error(exc))
            return

        calls = reply.tool_calls or []
        if not calls or not can_use_tools:  # no tool requested (or none allowed any more): this is the answer
            answer = reply.content or "(The model returned an empty answer.)"
            if getattr(choice, "finish_reason", None) == "length":
                answer += TRUNCATION_NOTE
            messages.append({"role": "assistant", "content": answer})
            yield AgentEvent(kind="final", step=step - 1, content=answer)
            return

        if reply.content:
            yield AgentEvent(kind="reasoning", step=step, content=reply.content)
        messages.append({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls],
        })
        for call in calls:
            yield AgentEvent(kind="tool_call", step=step, name=call.function.name, content=call.function.arguments)
            outcome = toolbox.call(call.function.name, call.function.arguments)
            yield AgentEvent(kind="tool_result", step=step, name=outcome.name, arguments=outcome.arguments,
                             content=outcome.summary, data=outcome.data, is_error=outcome.is_error)
            if outcome.memory_note:
                yield AgentEvent(kind="memory", step=step, name=outcome.name, content=outcome.memory_note)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})


def tool_log_entry(event: AgentEvent) -> str:
    """Compact JSON record of a tool_result event, stored with the assistant's chat message."""
    return json.dumps({"name": event.name, "arguments": event.arguments, "ok": not event.is_error})


def explain_llm_error(exc: Exception) -> str:
    kind = type(exc).__name__
    hints = {
        "AuthenticationError": "OpenAI rejected the API key. Check OPENAI_API_KEY.",
        "RateLimitError": "OpenAI rate limit or quota reached. Wait a moment or check your billing.",
        "APIConnectionError": "Could not reach OpenAI. Check your network connection.",
        "APITimeoutError": "The OpenAI request timed out. Try again.",
    }
    return hints.get(kind, f"LLM call failed ({kind}): {exc}")
