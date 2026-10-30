import shutil
import subprocess
from collections.abc import AsyncIterator

import pytest

from agent_smith.core.config import Settings
from agent_smith.core.errors import SandboxTimeoutError
from agent_smith.sandbox import DockerSandbox


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0  # noqa: S603, S607


pytestmark = pytest.mark.skipif(not _docker_ready(), reason="docker daemon not available")


@pytest.fixture
async def sandbox() -> AsyncIterator[DockerSandbox]:
    sb = DockerSandbox(Settings(sandbox_image="python:3.12-slim"))
    yield sb
    await sb.aclose()


async def test_runs_a_command(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("echo hello")
    assert result.ok and result.stdout.strip() == "hello"


async def test_exit_code_and_stderr(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("echo oops >&2; exit 3")
    assert result.exit_code == 3 and "oops" in result.stderr


async def test_workspace_persists_between_runs(sandbox: DockerSandbox) -> None:
    await sandbox.run("echo data > f.txt")
    assert (sandbox.workspace / "f.txt").read_text().strip() == "data"
    assert (await sandbox.run("cat f.txt")).stdout.strip() == "data"


async def test_root_filesystem_is_read_only(sandbox: DockerSandbox) -> None:
    assert not (await sandbox.run("touch /etc/x")).ok


async def test_no_network(sandbox: DockerSandbox) -> None:
    result = await sandbox.run(
        "python -c \"import urllib.request as u; u.urlopen('http://1.1.1.1', timeout=3)\""
    )
    assert not result.ok


async def test_host_env_is_not_leaked(
    sandbox: DockerSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert "secret" not in (await sandbox.run("env")).stdout


async def test_timeout_kills_the_container(sandbox: DockerSandbox) -> None:
    with pytest.raises(SandboxTimeoutError):
        await sandbox.run("sleep 30", timeout=2)
    ps = subprocess.run(  # noqa: S603, ASYNC221
        ["docker", "ps", "-q", "--filter", "name=agent-smith-"],  # noqa: S607
        capture_output=True,
        text=True,
    )
    assert ps.stdout.strip() == ""


async def test_output_is_truncated() -> None:
    sb = DockerSandbox(Settings(sandbox_max_output_bytes=100))
    try:
        result = await sb.run("yes | head -c 10000")
        assert result.truncated and len(result.stdout) == 100
    finally:
        await sb.aclose()
