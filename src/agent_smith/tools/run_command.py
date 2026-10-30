from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_smith.core.errors import SandboxError
from agent_smith.core.logging import get_logger
from agent_smith.sandbox import Sandbox, SandboxResult
from agent_smith.sandbox.policy import Approver, CommandPolicy, Verdict

log = get_logger(__name__)


class RunCommandInput(BaseModel):
    command: str = Field(description="Shell command to run in the sandbox workspace.")


def format_result(result: SandboxResult) -> str:
    parts = [f"exit code: {result.exit_code}"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    if result.truncated:
        parts.append("(output truncated)")
    return "\n".join(parts)


def make_run_command_tool(
    sandbox: Sandbox, policy: CommandPolicy, approver: Approver | None = None
) -> BaseTool:
    async def run_command(command: str) -> str:
        decision = policy.evaluate(command)
        log.info("run_command.policy", command=command, verdict=decision.verdict.value)
        if decision.verdict is Verdict.DENY:
            return f"Command blocked by policy: {decision.reason}."
        if decision.verdict is Verdict.ASK and not (
            approver is not None and await approver(command, decision.reason)
        ):
            return "Command was not approved, so it was not run."
        try:
            return format_result(await sandbox.run(command))
        except SandboxError as exc:
            return f"Could not run command: {exc}"

    return StructuredTool.from_function(
        coroutine=run_command,
        name="run_command",
        description=(
            "Run a shell command in an isolated sandbox with no network access. The working "
            "directory is the workspace, shared with write_file; nothing outside it persists. "
            "Returns the exit code, stdout and stderr."
        ),
        args_schema=RunCommandInput,
    )
