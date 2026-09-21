"""
Configuration management for GitHub Agent Studio.

Uses pydantic-settings to load environment variables and provide validated configuration.
All settings are accessible as attributes of the Config instance.
"""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables.
    
    Required:
        OPENAI_API_KEY: OpenAI API key for LLM access
        
    Optional:
        OPENAI_MODEL: LLM model to use (default: gpt-4o-mini)
        MAX_OUTPUT_TOKENS: Cap on tokens the LLM may generate per call (default: 2048)
        GITHUB_TOKEN: GitHub API token (optional but recommended)
    """
    
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    max_output_tokens: int = 2048
    github_token: str | None = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global config instance
config = Config()
