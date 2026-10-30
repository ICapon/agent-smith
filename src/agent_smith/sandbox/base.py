from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class SandboxResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    # True when stdout or stderr was cut at the configured output limit.
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Sandbox(Protocol):
    """Isolated environment where agent-generated commands are executed.

    Commands run against a persistent `workspace`: files written by one call are
    visible to the next. Nothing outside the workspace is writable.
    """

    @property
    def workspace(self) -> Path:
        """Host directory that is exposed to commands as the working directory."""
        ...

    async def run(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> SandboxResult:
        """Run a shell command, raising `SandboxTimeoutError` if it overruns `timeout`."""
        ...

    async def aclose(self) -> None:
        """Release resources, including the workspace if the sandbox created it."""
        ...
