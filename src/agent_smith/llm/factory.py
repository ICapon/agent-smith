from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import ConfigError


def create_llm(settings: Settings | None = None, **overrides: object) -> BaseChatModel:
    """Build the chat model from settings.

    `overrides` (e.g. `temperature=0.7`, `max_tokens=512`) take precedence over settings,
    so individual graph nodes can tune the model without touching global config.
    """
    settings = settings or get_settings()
    if settings.anthropic_api_key is None:
        raise ConfigError("ANTHROPIC_API_KEY is not set (env var or .env)")

    params: dict[str, object] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        **overrides,
    }
    return ChatAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        **params,
    )
