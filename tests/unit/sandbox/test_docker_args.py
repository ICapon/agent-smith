import asyncio
from pathlib import Path

from agent_smith.core.config import Settings
from agent_smith.sandbox.docker import DockerSandbox, _read_capped


def args_for(tmp_path: Path, **overrides: object) -> list[str]:
    sandbox = DockerSandbox(Settings(**overrides), workspace=tmp_path)  # type: ignore[arg-type]
    return sandbox._docker_args("c1", "echo hi")


def test_default_args_are_locked_down(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    assert args[:3] == ["docker", "run", "--rm"]
    assert args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert args[-4:] == ["python:3.12-slim", "sh", "-c", "echo hi"]


def test_workspace_is_the_only_bind_mount(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    volumes = [args[i + 1] for i, a in enumerate(args) if a == "--volume"]
    assert volumes == [f"{tmp_path.resolve()}:/workspace"]


def test_no_host_env_is_forwarded(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    envs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert envs == ["HOME=/tmp"]


def test_command_is_a_single_argument_after_the_image(tmp_path: Path) -> None:
    args = args_for(tmp_path, sandbox_image="img:1")
    assert args[-4:] == ["img:1", "sh", "-c", "echo hi"]


async def test_read_capped_truncates_but_drains() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"x" * 100_000)
    reader.feed_eof()
    data, truncated = await _read_capped(reader, 10)
    assert data == b"x" * 10 and truncated


async def test_read_capped_under_limit() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"abc")
    reader.feed_eof()
    assert await _read_capped(reader, 10) == (b"abc", False)


async def test_owned_workspace_is_removed_on_close() -> None:
    sandbox = DockerSandbox(Settings())
    assert sandbox.workspace.exists()
    await sandbox.aclose()
    assert not sandbox.workspace.exists()


async def test_given_workspace_is_kept_on_close(tmp_path: Path) -> None:
    await DockerSandbox(Settings(), workspace=tmp_path).aclose()
    assert tmp_path.exists()  # noqa: ASYNC240
