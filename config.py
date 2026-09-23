"""
Configuration management for GitHub Agent Studio.

Uses pydantic-settings to load environment variables and provide validated configuration.
All settings are accessible as attributes of the Config instance.

The ``.env`` file is looked up next to this file (the project root), not in the current
directory, so the app behaves the same wherever it is launched from.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent / ".env"


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Required:
        OPENAI_API_KEY: OpenAI API key for LLM access

    Optional:
        OPENAI_MODEL: LLM model to use (default: gpt-4o-mini)
        LITE_MODE: Cheaper test runs: smaller tool results and fewer tool rounds (default: off)
        MAX_OUTPUT_TOKENS: Cap on tokens the LLM may generate per call (default: 2048)
        GITHUB_TOKEN: GitHub API token (optional but recommended)
        TAVILY_API_KEY: Tavily search API key (optional; enables live CVE lookup for the Security Specialist)
        MEMORY_BACKEND: "sqlite" (default; persists to agent_memory.db) or "memory" (in-process only,
            forgotten on restart — for Streamlit Cloud, whose filesystem is itself not durable, so a
            SQLite file there would falsely look persistent)
    """

    model_config = SettingsConfigDict(env_file=ENV_FILE, case_sensitive=False, extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    max_output_tokens: int = Field(2048, ge=16, le=16_384)  # 16384 is gpt-4o-mini's own per-call output cap
    lite_mode: bool = False
    github_token: str | None = None
    tavily_api_key: str | None = None
    memory_backend: Literal["sqlite", "memory"] = "sqlite"


# Global config instance
config = Config()
