# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run the CLI (entry `agent_smith:main`, which re-exports `agent_smith.cli.main`):
  - Interactive research session: `uv run agent-smith`
  - One-shot research question: `uv run agent-smith "question"`
  - Coding agent in a Docker sandbox: `uv run agent-smith --code "task" [--workspace DIR]` (`--workspace` only valid with `--code`)
  - `-v` enables debug logging
- Tests: `uv run pytest`; single test: `uv run pytest tests/unit/agent/test_route.py::test_name`. `asyncio_mode = "auto"`, so async tests need no marker.
- Lint / format / types: `uv run ruff check .`, `uv run ruff format .`, `uv run mypy` (strict, pydantic plugin). Ruff line length is 100.
- Dependencies: `uv add <pkg>` (dev: `uv add --dev <pkg>`). Build: `uv build`.
- Python 3.12; `src/` layout with the `uv_build` backend.

## Configuration

`core/config.py` defines a frozen `Settings` (pydantic-settings) read from `AGENT_SMITH_<FIELD>` env vars and `.env` (see `.env.example`). The API key also accepts plain `ANTHROPIC_API_KEY`. `get_settings()` is `lru_cache`d. `temperature` defaults to `None` on purpose: `claude-sonnet-5-5` rejects any explicit value. All limits (iterations, queries, sandbox CPU/memory/timeouts, `coding_max_steps`) live here.

## Architecture

Two independent LangGraph agents, both built by factory functions that take their dependencies (LLM, scraper, search provider, sandbox) as arguments. This is what makes the unit tests able to inject fakes.

**Research graph** (`agent/graph.py`, nodes in `agent/nodes/`): `route → plan → search → scrape → synthesize`.
- `route` makes a structured-output LLM call (`research` / `chat` / `clarify`). Non-research decisions fill `answer` themselves and the graph ends right there (`route_after_route`).
- After `synthesize`, the graph loops back to `plan` only while `missing_info` is set and `iteration < max_iterations`. This is the only cycle.
- `AgentState` (`agent/state.py`) is the internal state. Callers only see `GraphInput` (`question`) and `GraphOutput` (`answer`, `sources`, `missing_info`). List fields that accumulate across iterations use `Annotated[..., add]` reducers.
- Nodes are `make_*_node(...)` closures returning async functions that return only the state fields they change (`AgentNode` Protocol). Prompts live in `agent/prompts.py`.
- The CLI sets the recursion limit to `4 * max_iterations + 5` (`_STEPS_PER_ITERATION` in `cli.py`).

**Coding agent** (`agent/coding.py`): a tool-calling loop (`agent ⇄ tools` via `ToolNode` and `tools_condition`) with `parallel_tool_calls=False`. Run it with `solve()`, which converts the step budget (`coding_max_steps`) into a LangGraph `recursion_limit` and raises `AgentLimitError` when it is exceeded.

**Sandbox** (`sandbox/`): the `Sandbox` Protocol (`base.py`) has a `DockerSandbox` implementation. Each command runs in a fresh `docker run --rm` container: no network, read-only root, `--cap-drop ALL`, non-root, CPU/memory/pid limits, no host env forwarded. Only the bind-mounted workspace persists between calls. The workspace is a temp dir unless the caller passes one.
- `CommandPolicy` (`policy.py`) returns allow / ask / deny. It is a convenience layer, not a security boundary: the container is the boundary. Only simple read-only commands with no shell metacharacters are auto-allowed; deny rules block obviously hostile ones; everything else needs approval.
- `tools/run_command.py` consults the policy. `ASK` goes to an injected `Approver`, which is `console_approver` in the CLI. With no approver, `ASK` is refused. `tools/write_file.py` writes into the same workspace, and `tools/__init__.py` has `build_sandbox_tools` (code tools) and `build_tools` (web search and scrape).

**Supporting packages**: `scraper/` (httpx fetcher with tenacity retries, robots.txt check, bs4/lxml extraction), `search/` (`SearchProvider` Protocol; `DuckDuckGoSearch`), `llm/factory.py` (`create_llm`, raises `ConfigError` on a missing key), `core/errors.py` (`AgentSmithError` hierarchy; the CLI maps `ConfigError` to exit 2, other errors to exit 1, Ctrl-C to 130), `core/logging.py` (structlog).

## Tests

`tests/unit/` mirrors `src/` and uses fakes and `respx`, with no network or Docker. `tests/integration/test_docker_sandbox.py` is skipped automatically when the Docker daemon is unavailable.
