from pathlib import Path

import pytest

from agent_smith.core.errors import WorkspacePathError
from agent_smith.sandbox.workspace import resolve_in_workspace


def test_relative_path_resolves_inside(tmp_path: Path) -> None:
    assert resolve_in_workspace(tmp_path, "a/b.py") == tmp_path.resolve() / "a" / "b.py"


@pytest.mark.parametrize("bad", ["", "  ", "/etc/passwd", "../x", "a/../../x", ".", "a\0b"])
def test_bad_paths_are_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(WorkspacePathError):
        resolve_in_workspace(tmp_path, bad)


def test_symlink_pointing_outside_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "link").symlink_to(outside)
    with pytest.raises(WorkspacePathError):
        resolve_in_workspace(ws, "link/pwned.txt")


def test_symlink_pointing_inside_is_fine(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    assert resolve_in_workspace(tmp_path, "alias/f") == tmp_path.resolve() / "real" / "f"
