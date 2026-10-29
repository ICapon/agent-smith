import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import SearchError
from agent_smith.core.logging import get_logger
from agent_smith.search.base import SearchProvider, SearchResult

log = get_logger(__name__)

SearchNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def make_search_node(provider: SearchProvider, settings: Settings | None = None) -> SearchNode:
    """Build the `search` node: runs every planned query and collects new results."""
    settings = settings or get_settings()

    async def run_query(query: str) -> list[SearchResult]:
        try:
            return await provider.search(query, settings.max_search_results)
        except SearchError as exc:
            # One failing query must not sink the others.
            log.warning("search_failed", query=query, error=str(exc))
            return []

    async def search(state: AgentState) -> dict[str, Any]:
        batches = await asyncio.gather(*(run_query(q) for q in state["queries"]))

        seen = {r.url for r in state.get("search_results", [])}
        new: list[SearchResult] = []
        for result in (r for batch in batches for r in batch):
            if result.url not in seen:
                seen.add(result.url)
                new.append(result)

        log.info("search", queries=len(state["queries"]), new_results=len(new))
        return {"search_results": new}

    return search
