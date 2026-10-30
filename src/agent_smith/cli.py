import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from agent_smith.agent.coding import CodingOutput, build_coding_agent, solve
from agent_smith.agent.graph import GraphOutput, build_graph
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import AgentSmithError, ConfigError
from agent_smith.core.logging import configure_logging
from agent_smith.llm import create_llm
from agent_smith.sandbox import DockerSandbox
from agent_smith.scraper import Scraper
from agent_smith.search import DuckDuckGoSearch
from agent_smith.tools import build_sandbox_tools

# Each pass visits 4 nodes; leave headroom over LangGraph's default recursion limit of 25.
_STEPS_PER_ITERATION = 4
_EXIT_COMMANDS = frozenset({"exit", "quit"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-smith", description="Answer a question by searching and reading the web."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="the question to research; omit to start an interactive session",
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="treat the text as a coding task and run the coding agent in a Docker sandbox",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="with --code: directory the agent works in and keeps (default: a temporary one)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


@asynccontextmanager
async def _open_graph(settings: Settings) -> AsyncIterator[Any]:
    llm = create_llm(settings)  # fail fast on config problems, before opening any client
    async with Scraper(settings) as scraper, DuckDuckGoSearch(settings) as provider:
        yield build_graph(llm, scraper, provider, settings)


async def _ask(graph: Any, question: str, settings: Settings) -> GraphOutput:
    result = await graph.ainvoke(
        {"question": question},
        config={"recursion_limit": _STEPS_PER_ITERATION * settings.max_iterations + 5},
    )
    return cast(GraphOutput, result)


async def run(question: str, settings: Settings) -> GraphOutput:
    async with _open_graph(settings) as graph:
        return await _ask(graph, question, settings)


async def chat(settings: Settings) -> None:
    """Interactive session: one graph and set of clients reused across questions.

    Each question is researched independently (no memory of earlier turns). A failure in
    one turn is reported and the session continues.
    """
    async with _open_graph(settings) as graph:
        print("agent-smith interactive session. Type 'exit' or 'quit' (or Ctrl-D) to leave.")
        while True:
            try:
                line = await asyncio.to_thread(input, "\n> ")
            except EOFError:
                print()
                return
            question = line.strip()
            if not question:
                continue
            if question.lower() in _EXIT_COMMANDS:
                return
            try:
                print(render(await _ask(graph, question, settings)))
            except AgentSmithError as exc:
                print(f"Error: {exc}", file=sys.stderr)


async def console_approver(command: str, reason: str) -> bool:
    """Ask the person at the terminal before a non-allowlisted command runs."""
    prompt = f"\nThe agent wants to run in the sandbox:\n  {command}\n({reason}) Allow? [y/N] "
    try:
        reply = await asyncio.to_thread(input, prompt)
    except EOFError:
        return False
    return reply.strip().lower() in {"y", "yes", "s", "si", "sí"}


async def run_code(task: str, settings: Settings, workspace: Path | None = None) -> CodingOutput:
    llm = create_llm(settings)  # fail fast on config problems, before creating a sandbox
    sandbox = DockerSandbox(settings, workspace=workspace)
    try:
        tools = build_sandbox_tools(sandbox, approver=console_approver, settings=settings)
        agent = build_coding_agent(llm, tools, settings)
        return await solve(agent, task, settings)
    finally:
        await sandbox.aclose()


def render(output: GraphOutput) -> str:
    lines = [output["answer"]]
    if output["sources"]:
        lines += ["", "Sources:", *(f"  [{i}] {url}" for i, url in enumerate(output["sources"], 1))]
    if output["missing_info"]:
        lines += ["", f"Note: the answer may be incomplete. Missing: {output['missing_info']}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.verbose:
        settings = settings.model_copy(update={"log_level": "DEBUG"})
    configure_logging(settings)

    if args.code and not args.question:
        print(
            "--code needs a task, e.g.: agent-smith --code 'write fizzbuzz and run it'",
            file=sys.stderr,
        )
        return 2
    if args.workspace and not args.code:
        print("--workspace only applies with --code", file=sys.stderr)
        return 2

    try:
        if args.code:
            print(
                asyncio.run(run_code(" ".join(args.question), settings, args.workspace))["answer"]
            )
            return 0
        if not args.question:
            asyncio.run(chat(settings))
            return 0
        output = asyncio.run(run(" ".join(args.question), settings))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except AgentSmithError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    print(render(output))
    return 0
