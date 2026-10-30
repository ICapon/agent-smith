from pathlib import Path

from agent_smith.core.config import Settings
from agent_smith.core.errors import SandboxTimeoutError, SandboxUnavailableError
from agent_smith.sandbox import CommandPolicy, SandboxResult
from agent_smith.tools import build_sandbox_tools


class FakeSandbox:
    def __init__(self, workspace: Path, result: SandboxResult | Exception | None = None) -> None:
        self._workspace = workspace
        self.result = result or SandboxResult(exit_code=0, stdout="ok\n")
        self.ran: list[str] = []

    @property
    def workspace(self) -> Path:
        return self._workspace

    async def run(self, command: str, *, timeout: float | None = None) -> SandboxResult:  # noqa: ASYNC109
        self.ran.append(command)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def aclose(self) -> None: ...


def tools(sandbox: FakeSandbox, approver=None, **settings: int):  # type: ignore[no-untyped-def]
    built = build_sandbox_tools(
        sandbox,
        CommandPolicy(),
        approver,
        Settings(_env_file=None, **settings),  # type: ignore[arg-type]
    )
    return {t.name: t for t in built}


async def test_allowed_command_runs_without_approval(tmp_path: Path) -> None:
    sb = FakeSandbox(tmp_path)
    out = await tools(sb)["run_command"].ainvoke({"command": "ls"})
    assert sb.ran == ["ls"] and out == "exit code: 0\nstdout:\nok\n"


async def test_denied_command_never_reaches_the_sandbox(tmp_path: Path) -> None:
    sb = FakeSandbox(tmp_path)

    async def approve_all(command: str, reason: str) -> bool:
        return True

    out = await tools(sb, approve_all)["run_command"].ainvoke({"command": "sudo ls"})
    assert sb.ran == [] and "blocked by policy" in out


async def test_ask_without_approver_fails_closed(tmp_path: Path) -> None:
    sb = FakeSandbox(tmp_path)
    out = await tools(sb)["run_command"].ainvoke({"command": "python x.py"})
    assert sb.ran == [] and "not approved" in out


async def test_ask_runs_only_when_approved(tmp_path: Path) -> None:
    asked: list[tuple[str, str]] = []

    async def approver(command: str, reason: str) -> bool:
        asked.append((command, reason))
        return command == "python yes.py"

    sb = FakeSandbox(tmp_path)
    t = tools(sb, approver)["run_command"]
    assert "not approved" in await t.ainvoke({"command": "python no.py"})
    assert "exit code: 0" in await t.ainvoke({"command": "python yes.py"})
    assert sb.ran == ["python yes.py"] and len(asked) == 2


async def test_failure_output_is_formatted(tmp_path: Path) -> None:
    result = SandboxResult(exit_code=2, stderr="boom", truncated=True)
    out = await tools(FakeSandbox(tmp_path, result))["run_command"].ainvoke({"command": "ls"})
    assert out == "exit code: 2\nstderr:\nboom\n(output truncated)"


async def test_sandbox_errors_become_tool_output(tmp_path: Path) -> None:
    for exc in (SandboxTimeoutError("ls", 1), SandboxUnavailableError("no docker")):
        sb = FakeSandbox(tmp_path, exc)
        out = await tools(sb)["run_command"].ainvoke({"command": "ls"})
        assert out.startswith("Could not run command:")


async def test_write_file_creates_parents(tmp_path: Path) -> None:
    out = await tools(FakeSandbox(tmp_path))["write_file"].ainvoke(
        {"path": "src/a.py", "content": "print(1)\n"}
    )
    assert out == "Wrote 9 bytes to src/a.py."
    assert (tmp_path / "src" / "a.py").read_text() == "print(1)\n"  # noqa: ASYNC240


async def test_write_file_rejects_escapes(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    t = tools(FakeSandbox(ws))["write_file"]
    for path in ("../evil.txt", "/tmp/evil.txt"):
        out = await t.ainvoke({"path": path, "content": "x"})
        assert out.startswith("Could not write file:")
    assert not (tmp_path / "evil.txt").exists()  # noqa: ASYNC240


async def test_write_file_does_not_follow_symlink_out(tmp_path: Path) -> None:
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "link").symlink_to(outside)
    out = await tools(FakeSandbox(ws))["write_file"].ainvoke(
        {"path": "link/evil.txt", "content": "x"}
    )
    assert out.startswith("Could not write file:")
    assert not (outside / "evil.txt").exists()  # noqa: ASYNC240


async def test_write_file_enforces_size_limit(tmp_path: Path) -> None:
    out = await tools(FakeSandbox(tmp_path), sandbox_max_file_bytes=5)["write_file"].ainvoke(
        {"path": "big.txt", "content": "123456"}
    )
    assert out.startswith("File too large")
    assert not (tmp_path / "big.txt").exists()  # noqa: ASYNC240


async def test_write_file_on_a_directory_reports_os_error(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    out = await tools(FakeSandbox(tmp_path))["write_file"].ainvoke({"path": "d", "content": "x"})
    assert out.startswith("Could not write file:")
