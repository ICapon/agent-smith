from typing import Any, cast

from agent_smith.agent.nodes.search import SearchNode, make_search_node
from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings
from agent_smith.core.errors import SearchError
from agent_smith.search.base import SearchResult


def result(n: int) -> SearchResult:
    return SearchResult(title=f"T{n}", url=f"https://e.com/{n}")


class FakeProvider:
    def __init__(self, by_query: dict[str, list[SearchResult] | Exception]) -> None:
        self.by_query = by_query

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        outcome = self.by_query[query]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome[:max_results]

    async def aclose(self) -> None: ...


def node(provider: FakeProvider, **kw: Any) -> SearchNode:
    return make_search_node(provider, Settings(_env_file=None, **kw))


async def test_merges_and_dedupes_across_queries() -> None:
    provider = FakeProvider({"a": [result(1), result(2)], "b": [result(2), result(3)]})
    out = await node(provider)(cast(AgentState, {"queries": ["a", "b"]}))
    assert [r.url[-1] for r in out["search_results"]] == ["1", "2", "3"]


async def test_skips_results_already_in_state() -> None:
    provider = FakeProvider({"a": [result(1), result(2)]})
    state = cast(AgentState, {"queries": ["a"], "search_results": [result(1)]})
    out = await node(provider)(state)
    assert [r.url[-1] for r in out["search_results"]] == ["2"]


async def test_failed_query_does_not_sink_others() -> None:
    provider = FakeProvider({"a": SearchError("blocked"), "b": [result(1)]})
    out = await node(provider)(cast(AgentState, {"queries": ["a", "b"]}))
    assert len(out["search_results"]) == 1


async def test_respects_max_results_per_query() -> None:
    provider = FakeProvider({"a": [result(i) for i in range(10)]})
    out = await node(provider, max_search_results=2)(cast(AgentState, {"queries": ["a"]}))
    assert len(out["search_results"]) == 2
