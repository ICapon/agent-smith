# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Freshly scaffolded `uv` project (`uv init --package` layout): a single stub `main()` in `src/agent_smith/__init__.py`, an empty `README.md`, no dependencies, no tests, and no linter configured. Update this file as real architecture, tooling and test setup are added.

## Commands

- Run the CLI entry point: `uv run agent-smith` (maps to `agent_smith:main` via `[project.scripts]`)
- Add a dependency: `uv add <pkg>` (dev: `uv add --dev <pkg>`)
- Build: `uv build` (backend is `uv_build`)

## Notes

- Python 3.12 (`.python-version`, `requires-python >=3.12`).
- `src/` layout: the package lives at `src/agent_smith/`; the `uv_build` backend expects this module name to match the project name.
