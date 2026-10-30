from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from agent_smith.agent.coding import build_coding_agent, solve
from agent_smith.core.config import Settings
from agent_smith.core.errors import AgentLimitError
from agent_smith.sandbox import CommandPolicy, SandboxResult
from agent_smith.tools import build_sandbox_tools


def call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedLLM:
    """Replays queued AI messages and records what it was shown on each turn."""

    def __init__(self, *replies: AIMessage, repeat_last: bool = False) -> None:
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.seen: list[list[BaseMessage]] = []
        self.bound_kwargs: dict[str, Any] = {}

    def bind_tools(self, tools: Any, **kwargs: Any) -> RunnableLambda[Any, AIMessage]:
        self.bound_kwargs = kwargs

        def reply(prompt: Any) -> AIMessage:
            self.seen.append(prompt.to_messages())
            # A copy, so add_messages sees a new message rather than an update to an old id.
            return self.replies[0].model_copy() if self.repeat_last else self.replies.pop(0)

        return RunnableLambda(reply)


class FakeSandbox:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.ran: list[str] = []

    async def run(self, command: str, *, timeout: float | None = None) -> SandboxResult:  # noqa: ASYNC109
        self.ran.append(command)
        return SandboxResult(exit_code=0, stdout="42\n")

    async def aclose(self) -> None: ...


async def approve_all(command: str, reason: str) -> bool:
    return True


def make(llm: ScriptedLLM, tmp_path: Path, **settings: Any) -> tuple[Any, FakeSandbox, Settings]:
    cfg = Settings(_env_file=None, **settings)
    sandbox = FakeSandbox(tmp_path)
    tools = build_sandbox_tools(sandbox, CommandPolicy(), approve_all, cfg)
    agent = build_coding_agent(cast(BaseChatModel, llm), tools, cfg)
    return agent, sandbox, cfg


async def test_writes_runs_and_answers(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        call("write_file", "1", path="a.py", content="print(6*7)\n"),
        call("run_command", "2", command="python a.py"),
        AIMessage(content=" The result is 42. "),
    )
    agent, sandbox, cfg = make(llm, tmp_path)

    out = await solve(agent, "compute 6*7", cfg)

    assert out == {"answer": "The result is 42."}
    assert (tmp_path / "a.py").read_text() == "print(6*7)\n"  # noqa: ASYNC240
    assert sandbox.ran == ["python a.py"]
    # The model saw the first tool result before choosing its second step.
    results = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == ["1", "2"]
    assert "Wrote 11 bytes" in str(results[0].content) and "42" in str(results[1].content)
    assert llm.bound_kwargs == {"parallel_tool_calls": False}


async def test_task_reaches_the_model_as_a_human_message(tmp_path: Path) -> None:
    llm = ScriptedLLM(AIMessage(content="done"))
    agent, _, cfg = make(llm, tmp_path)
    await solve(agent, "do the thing", cfg)
    first_turn = llm.seen[0]
    assert first_turn[0].type == "system" and first_turn[-1].content == "do the thing"


async def test_blocked_command_is_reported_back_to_the_model(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        call("run_command", "1", command="sudo ls"),
        AIMessage(content="I could not do that."),
    )
    agent, sandbox, cfg = make(llm, tmp_path)

    out = await solve(agent, "task", cfg)

    assert out["answer"] == "I could not do that."
    assert sandbox.ran == []
    tool_msgs = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert "blocked by policy" in str(tool_msgs[0].content)


async def test_invalid_tool_arguments_do_not_crash_the_loop(tmp_path: Path) -> None:
    llm = ScriptedLLM(call("write_file", "1", path="a.py"), AIMessage(content="recovered"))
    agent, _, cfg = make(llm, tmp_path)
    assert (await solve(agent, "task", cfg))["answer"] == "recovered"


async def test_empty_final_reply_has_a_fallback(tmp_path: Path) -> None:
    agent, _, cfg = make(ScriptedLLM(AIMessage(content="")), tmp_path)
    assert (await solve(agent, "task", cfg))["answer"] == "(no answer)"


async def test_step_budget_stops_a_model_that_never_finishes(tmp_path: Path) -> None:
    llm = ScriptedLLM(call("run_command", "1", command="ls"), repeat_last=True)
    agent, sandbox, cfg = make(llm, tmp_path, coding_max_steps=2)
    with pytest.raises(AgentLimitError, match="2 tool rounds"):
        await solve(agent, "task", cfg)
    assert len(sandbox.ran) == 2
