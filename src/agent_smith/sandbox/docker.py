import asyncio
import contextlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import SandboxTimeoutError, SandboxUnavailableError
from agent_smith.core.logging import get_logger
from agent_smith.sandbox.base import SandboxResult

log = get_logger(__name__)

_CONTAINER_WORKDIR = "/workspace"


class DockerSandbox:
    """Runs each command in a fresh, locked-down, throwaway Docker container.

    Isolation comes from the container flags, not from trusting the command: no
    network by default, read-only root filesystem, no capabilities, non-root user,
    and CPU/memory/pid limits. The only writable paths are `/tmp` (tmpfs) and the
    workspace bind mount. No environment variables from the host are forwarded.
    """

    def __init__(self, settings: Settings | None = None, workspace: Path | None = None) -> None:
        self._settings = settings or get_settings()
        self._owns_workspace = workspace is None
        self._workspace = (
            workspace.resolve()
            if workspace is not None
            else Path(tempfile.mkdtemp(prefix="agent-smith-")).resolve()
        )
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def _docker_args(self, name: str, command: str) -> list[str]:
        s = self._settings
        args = [
            "docker", "run", "--rm", "-i",
            "--name", name,
            "--network", s.sandbox_network,
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # noqa: S108
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", s.sandbox_memory,
            "--cpus", str(s.sandbox_cpus),
            "--pids-limit", str(s.sandbox_pids_limit),
            "--env", "HOME=/tmp",
            "--volume", f"{self._workspace}:{_CONTAINER_WORKDIR}",
            "--workdir", _CONTAINER_WORKDIR,
        ]  # fmt: skip
        if hasattr(os, "getuid"):
            # Keep files in the workspace owned by the host user (matters on Linux).
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return [*args, s.sandbox_image, "sh", "-c", command]

    async def run(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> SandboxResult:
        timeout = timeout if timeout is not None else self._settings.sandbox_timeout
        name = f"agent-smith-{uuid.uuid4().hex[:12]}"
        limit = self._settings.sandbox_max_output_bytes
        log.info("sandbox.run", container=name, command=command, timeout=timeout)

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._docker_args(name, command),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError("the `docker` CLI was not found on PATH") from exc

        assert proc.stdout is not None and proc.stderr is not None  # noqa: S101
        try:
            async with asyncio.timeout(timeout):
                (out, out_cut), (err, err_cut), code = await asyncio.gather(
                    _read_capped(proc.stdout, limit),
                    _read_capped(proc.stderr, limit),
                    proc.wait(),
                )
        except TimeoutError:
            await self._kill(name, proc)
            raise SandboxTimeoutError(command, timeout) from None
        except BaseException:
            await self._kill(name, proc)
            raise

        stderr = err.decode(errors="replace")
        if code == 125 and not out:
            # 125 = the docker CLI itself failed (daemon down, image missing, bad flag).
            raise SandboxUnavailableError(stderr.strip() or "docker run failed")

        log.info("sandbox.done", container=name, exit_code=code)
        return SandboxResult(
            exit_code=code,
            stdout=out.decode(errors="replace"),
            stderr=stderr,
            truncated=out_cut or err_cut,
        )

    async def _kill(self, name: str, proc: asyncio.subprocess.Process) -> None:
        # Killing the CLI would leave the container running, so kill the container.
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                "docker", "kill", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )  # fmt: skip
            await killer.wait()
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()

    async def aclose(self) -> None:
        if self._owns_workspace:
            await asyncio.to_thread(shutil.rmtree, self._workspace, ignore_errors=True)


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Read to EOF, keeping at most `limit` bytes and draining the rest.

    Draining matters: if we stopped reading, the child would block on a full pipe.
    """
    buf = bytearray()
    truncated = False
    while chunk := await stream.read(65536):
        room = limit - len(buf)
        if room > 0:
            buf += chunk[:room]
        if len(chunk) > room:
            truncated = True
    return bytes(buf), truncated
