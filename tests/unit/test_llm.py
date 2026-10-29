import pytest
from langchain_anthropic import ChatAnthropic

from agent_smith.core.config import Settings
from agent_smith.core.errors import ConfigError
from agent_smith.llm import create_llm


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        create_llm(Settings(_env_file=None))


def test_builds_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = create_llm(Settings(_env_file=None, model="claude-haiku-5-5", llm_max_retries=1))
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-haiku-5-5"
    assert llm.temperature is None
    assert llm.max_retries == 1


def test_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = create_llm(Settings(_env_file=None), temperature=0.7, max_tokens=100)
    assert isinstance(llm, ChatAnthropic)
    assert (llm.temperature, llm.max_tokens) == (0.7, 100)
