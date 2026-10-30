from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from agent_smith.agent.graph import build_graph
from agent_smith.agent.nodes.plan import QueryPlan
from agent_smith.agent.nodes.route import Route
from agent_smith.agent.nodes.synthesize import Synthesis
from agent_smith.core.config import Settings
from agent_smith.scraper.models import PageContent
from agent_smith.search.base import SearchResult


class ScriptedLLM:
    """Replays queued responses, one queue per structured-output schema."""

    def __init__(
        self, plans: list[QueryPlan], syntheses: list[Synthesis], routes: list[Route]
    ) -> None:
        self.queues: dict[type[BaseModel], list[BaseModel]] = {
            Route: list(routes),
            QueryPlan: list(plans),
            Synthesis: list(syntheses),
        }

    def with_structured_output(
        self, schema: type[BaseModel], **_: Any
    ) -> RunnableLambda[Any, BaseModel]:
        return RunnableLambda(lambda _: self.queues[schema].pop(0))


class FakeProvider:
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [SearchResult(title=query, url=f"https://e.com/{query}")]

    async def aclose(self) -> None: ...


class FakeScraper:
    async def scrape(self, url: str) -> PageContent:
        return PageContent(url=url, title=url, text=f"content of {url}")


def run(
    plans: list[QueryPlan],
    syntheses: list[Synthesis],
    routes: list[Route] | None = None,
    **settings: Any,
) -> Any:
    routes = routes or [Route(decision="research")]
    llm = cast(BaseChatModel, ScriptedLLM(plans, syntheses, routes))
    graph = build_graph(llm, FakeScraper(), FakeProvider(), Settings(_env_file=None, **settings))
    return graph.ainvoke({"question": "q?"})


async def test_single_pass_returns_only_output_fields() -> None:
    out = await run([QueryPlan(queries=["a"])], [Synthesis(answer="done [1]", cited_sources=[1])])
    assert out == {"answer": "done [1]", "sources": ["https://e.com/a"], "missing_info": None}


async def test_loops_back_while_info_is_missing() -> None:
    out = await run(
        [QueryPlan(queries=["a"]), QueryPlan(queries=["b"])],
        [
            Synthesis(answer="partial", cited_sources=[1], missing_info="need b"),
            Synthesis(answer="full", cited_sources=[1, 2]),
        ],
    )
    assert out["answer"] == "full"
    assert out["sources"] == ["https://e.com/a", "https://e.com/b"]  # pages accumulated
    assert out["missing_info"] is None


async def test_loop_is_bounded_by_max_iterations() -> None:
    gap = Synthesis(answer="still partial", missing_info="more")
    out = await run([QueryPlan(queries=[q]) for q in "abc"], [gap, gap, gap], max_iterations=2)
    assert out["missing_info"] == "more"  # stopped after 2 passes; the 3rd was never consumed


async def test_chat_route_replies_without_researching() -> None:
    out = await run([], [], routes=[Route(decision="chat", reply="Hi!")])
    assert out == {"answer": "Hi!", "sources": [], "missing_info": None}


async def test_clarify_route_asks_the_user() -> None:
    out = await run([], [], routes=[Route(decision="clarify", reply="About what?")])
    assert out["answer"] == "About what?"


async def test_empty_reply_falls_back_to_research() -> None:
    out = await run(
        [QueryPlan(queries=["a"])],
        [Synthesis(answer="done [1]", cited_sources=[1])],
        routes=[Route(decision="chat", reply="  ")],
    )
    assert out["answer"] == "done [1]"
