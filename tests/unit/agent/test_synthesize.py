from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from agent_smith.agent.nodes.synthesize import Synthesis, make_synthesize_node
from agent_smith.agent.state import AgentState
from agent_smith.scraper.models import PageContent


class StubLLM:
    def __init__(self, response: Synthesis) -> None:
        self.response = response
        self.seen: list[Any] = []

    def with_structured_output(self, schema: type[Synthesis], **_: Any) -> RunnableLambda[Any, Synthesis]:
        def respond(prompt: Any) -> Synthesis:
            self.seen.append(prompt)
            return self.response

        return RunnableLambda(respond)


def page(n: int) -> PageContent:
    return PageContent(url=f"https://e.com/{n}", title=f"T{n}", text=f"text{n}")


def run(stub: StubLLM, **state: Any) -> Any:
    node = make_synthesize_node(cast(BaseChatModel, stub))
    return node(cast(AgentState, {"question": "q?", **state}))


async def test_maps_citations_to_urls() -> None:
    stub = StubLLM(Synthesis(answer=" Yes [2][1]. ", cited_sources=[2, 1, 2], missing_info=None))
    out = await run(stub, pages=[page(1), page(2), page(3)])
    assert out == {
        "answer": "Yes [2][1].",
        "sources": ["https://e.com/1", "https://e.com/2"],
        "missing_info": None,
    }
    assert 'id="3"' in str(stub.seen[0].to_messages()[1].content)


async def test_drops_invented_citation_ids() -> None:
    stub = StubLLM(Synthesis(answer="a", cited_sources=[0, 1, 7, -1]))
    out = await run(stub, pages=[page(1)])
    assert out["sources"] == ["https://e.com/1"]


async def test_reports_missing_info_and_blank_means_complete() -> None:
    gap = await run(StubLLM(Synthesis(answer="a", missing_info=" the date ")), pages=[page(1)])
    assert gap["missing_info"] == "the date"
    done = await run(StubLLM(Synthesis(answer="a", missing_info="  ")), pages=[page(1)])
    assert done["missing_info"] is None


async def test_no_pages_skips_the_llm() -> None:
    stub = StubLLM(Synthesis(answer="should not be used"))
    out = await run(stub)
    assert stub.seen == []
    assert out["sources"] == [] and out["missing_info"]
