from pathlib import Path, PurePath

from agent_smith.core.errors import WorkspacePathError


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Map an agent-supplied relative path to a host path that is inside `workspace`.

    Symlinks are resolved before the containment check. This matters because commands
    run inside the container and can create a symlink in the workspace pointing anywhere;
    the host would otherwise follow it when writing.
    """
    if not relative.strip() or "\0" in relative:
        raise WorkspacePathError("path is empty or contains a NUL byte")
    if PurePath(relative).is_absolute():
        raise WorkspacePathError(f"{relative}: absolute paths are not allowed")
    root = workspace.resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise WorkspacePathError(f"{relative}: path escapes the workspace")
    return target
