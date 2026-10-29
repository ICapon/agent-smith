import pytest
from pydantic import ValidationError

from agent_smith.core.config import Settings
from agent_smith.core.errors import AgentSmithError, FetchError, RobotsDisallowedError
from agent_smith.core.logging import configure_logging, get_logger


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.anthropic_api_key is None
    assert s.respect_robots is True


def test_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_SMITH_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENT_SMITH_HTTP_TIMEOUT", "3.5")
    s = Settings(_env_file=None)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(s)
    assert s.log_level == "DEBUG"
    assert s.http_timeout == 3.5


def test_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SMITH_HTTP_TIMEOUT", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_errors_hierarchy() -> None:
    err = FetchError("http://x", "boom", status_code=503)
    assert isinstance(err, AgentSmithError)
    assert err.status_code == 503
    assert "disallowed" in str(RobotsDisallowedError("http://x"))


def test_logging_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(_env_file=None, log_json=True))
    get_logger("t").info("hello", k=1)
    assert '"k": 1' in capsys.readouterr().err
