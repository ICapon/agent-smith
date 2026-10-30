from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from agent_smith import cli
from agent_smith.agent.graph import GraphOutput
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import AgentSmithError


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


def test_chat_loops_until_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeGraph:
        async def ainvoke(self, state: dict[str, str], config: dict[str, int]) -> GraphOutput:
            if state["question"] == "boom":
                raise AgentSmithError("kaboom")
            return GraphOutput(answer=f"echo: {state['question']}", sources=[], missing_info=None)

    @asynccontextmanager
    async def fake_open_graph(settings: Settings) -> AsyncIterator[FakeGraph]:
        yield FakeGraph()

    inputs = iter(["first", "", "boom", "second", "EXIT", "never reached"])
    monkeypatch.setattr(cli, "_open_graph", fake_open_graph)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))

    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "echo: first" in captured.out and "echo: second" in captured.out
    assert "never reached" not in captured.out
    assert "kaboom" in captured.err
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("reply", "expected"), [("y", True), (" YES ", True), ("sí", True), ("", False), ("n", False)]
)
async def test_console_approver_reads_the_answer(
    monkeypatch: pytest.MonkeyPatch, reply: str, expected: bool
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: reply)
    assert await cli.console_approver("python x.py", "not on the allowlist") is expected


async def test_console_approver_denies_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert await cli.console_approver("python x.py", "r") is False


def test_main_code_requires_a_task(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--code"]) == 2
    assert "--code needs a task" in capsys.readouterr().err


def test_main_workspace_requires_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--workspace", "/tmp/w", "hello"]) == 2  # noqa: S108
    assert "only applies with --code" in capsys.readouterr().err


def test_main_code_runs_the_coding_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}

    async def fake_run_code(task: str, settings: Settings, workspace: object) -> dict[str, str]:
        seen.update(task=task, workspace=workspace)
        return {"answer": "42"}

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "run_code", fake_run_code)
    assert cli.main(["--code", "compute", "6*7"]) == 0
    assert capsys.readouterr().out.strip() == "42"
    assert seen == {"task": "compute 6*7", "workspace": None}
