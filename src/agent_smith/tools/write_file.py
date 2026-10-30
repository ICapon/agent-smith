import asyncio
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_smith.core.errors import SandboxError
from agent_smith.sandbox import Sandbox
from agent_smith.sandbox.workspace import resolve_in_workspace


class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the workspace, e.g. `src/main.py`.")
    content: str = Field(description="Full text content of the file; replaces any existing file.")


def make_write_file_tool(sandbox: Sandbox, max_bytes: int) -> BaseTool:
    async def write_file(path: str, content: str) -> str:
        data = content.encode()
        if len(data) > max_bytes:
            return f"File too large: {len(data)} bytes (limit {max_bytes})."
        try:
            target = resolve_in_workspace(sandbox.workspace, path)
            await asyncio.to_thread(_write, target, data)
        except SandboxError as exc:
            return f"Could not write file: {exc}"
        except OSError as exc:
            return f"Could not write file: {path}: {exc.strerror or exc}"
        return f"Wrote {len(data)} bytes to {path}."

    return StructuredTool.from_function(
        coroutine=write_file,
        name="write_file",
        description=(
            "Create or overwrite a text file in the sandbox workspace, creating parent "
            "directories as needed. Use run_command to execute it."
        ),
        args_schema=WriteFileInput,
    )


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
