"""
Thin LLM provider abstraction.

Every agent talks to the model through ``LLMProvider.chat()``, never through a raw
SDK client. Today the only implementation is ``OpenAIProvider``, a one-method wrapper
around the OpenAI client already used everywhere. Nothing about the tool-calling loop
is OpenAI-specific beyond this file: a second provider (Anthropic, a local model, ...)
plugs in here without touching ``agents/loop.py`` or any agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    """One chat call, OpenAI-compatible in and out: this codebase reads ``.choices[0].message``
    and ``.usage`` off whatever ``chat()`` returns, so a new provider's response must offer those,
    or the response needs adapting to that shape."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Any:
        """Send one chat-completion request and return the raw response object."""
        ...


class OpenAIProvider(LLMProvider):
    """Wraps an already-constructed OpenAI client; ``resolve_llm`` builds one when nothing is injected."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def chat(self, messages, *, tools=None, max_tokens=2048, temperature=0.2):
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools is not None else {}
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens, **kwargs
        )
