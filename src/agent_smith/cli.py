import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import cast

from agent_smith.agent.graph import GraphOutput, build_graph
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import AgentSmithError, ConfigError
from agent_smith.core.logging import configure_logging
from agent_smith.llm import create_llm
from agent_smith.scraper import Scraper
from agent_smith.search import DuckDuckGoSearch

# Each pass visits 4 nodes; leave headroom over LangGraph's default recursion limit of 25.
_STEPS_PER_ITERATION = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-smith", description="Answer a question by searching and reading the web."
    )
    parser.add_argument("question", nargs="+", help="the question to research")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


async def run(question: str, settings: Settings) -> GraphOutput:
    llm = create_llm(settings)  # fail fast on config problems, before opening any client
    async with Scraper(settings) as scraper, DuckDuckGoSearch(settings) as provider:
        graph = build_graph(llm, scraper, provider, settings)
        result = await graph.ainvoke(
            {"question": question},
            config={"recursion_limit": _STEPS_PER_ITERATION * settings.max_iterations + 5},
        )
        return cast(GraphOutput, result)


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

    try:
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
