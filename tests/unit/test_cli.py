import pytest

from agent_smith import cli
from agent_smith.agent.graph import GraphOutput
from agent_smith.core.config import Settings, get_settings


def test_parse_args_joins_unquoted_words() -> None:
    args = cli.parse_args(["what", "is", "langgraph?", "-v"])
    assert args.question == ["what", "is", "langgraph?"] and args.verbose


def test_render_full_output() -> None:
    out = GraphOutput(answer="Yes [1].", sources=["https://a", "https://b"], missing_info="date")
    text = cli.render(out)
    assert text.startswith("Yes [1].")
    assert "[2] https://b" in text and "Missing: date" in text


def test_render_minimal_output() -> None:
    assert cli.render(GraphOutput(answer="ok", sources=[], missing_info=None)) == "ok"


def test_main_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["hello"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_main_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run(question: str, settings: Settings) -> GraphOutput:
        return GraphOutput(answer=f"echo: {question}", sources=[], missing_info=None)

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main(["a", "b"]) == 0
    assert capsys.readouterr().out.strip() == "echo: a b"
    get_settings.cache_clear()
