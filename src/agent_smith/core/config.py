from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and `.env`.

    Everything is read from `AGENT_SMITH_<FIELD>` except the API key, which also
    accepts the standard `ANTHROPIC_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_SMITH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # LLM
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "AGENT_SMITH_ANTHROPIC_API_KEY"),
    )
    model: str = "claude-sonnet-5-5"
    # None = provider default; some models (e.g. claude-sonnet-5-5) reject any other value.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    # Scraper
    user_agent: str = "agent-smith/0.1 (+https://github.com/ICapon/agent-smith)"
    http_timeout: float = Field(default=15.0, gt=0)
    http_max_retries: int = Field(default=3, ge=0)
    max_concurrent_requests: int = Field(default=5, gt=0)
    max_response_bytes: int = Field(default=5_000_000, gt=0)
    max_content_chars: int = Field(default=20_000, gt=0)
    respect_robots: bool = True

    # Agent
    max_iterations: int = Field(default=3, gt=0)
    max_queries: int = Field(default=3, gt=0)
    max_search_results: int = Field(default=5, gt=0)
    max_pages_per_iteration: int = Field(default=3, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
