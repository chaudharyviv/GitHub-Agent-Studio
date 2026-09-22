"""Token and cost accounting for LLM calls, so the price of a run is visible instead of guessed."""

from dataclasses import dataclass
from typing import Any

# gpt-4o-mini list prices, USD per 1M tokens. Estimates only: check OpenAI's pricing page.
PRICE_INPUT = 0.15
PRICE_CACHED_INPUT = 0.075  # OpenAI caches repeated prompt prefixes automatically
PRICE_OUTPUT = 0.60


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    def add(self, response: Any) -> None:
        """Record one chat-completions response (ignored if it carries no usage data)."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached_tokens += (getattr(details, "cached_tokens", 0) or 0) if details else 0

    @property
    def cost_usd(self) -> float:
        fresh = self.prompt_tokens - self.cached_tokens
        return (fresh * PRICE_INPUT + self.cached_tokens * PRICE_CACHED_INPUT + self.completion_tokens * PRICE_OUTPUT) / 1_000_000

    def summary(self) -> str:
        return (f"{self.calls} LLM call(s) · {self.prompt_tokens:,} input tokens ({self.cached_tokens:,} cached) · "
                f"{self.completion_tokens:,} output · ≈ {self.cost_usd * 100:.2f}¢")
