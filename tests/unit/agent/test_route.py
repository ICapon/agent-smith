from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from agent_smith.agent.nodes.route import Route, make_route_node
from agent_smith.agent.state import AgentState


class StubLLM:
    def __init__(self, route: Route) -> None:
        self.route = route

    def with_structured_output(self, schema: type[Route], **_: Any) -> RunnableLambda[Any, Route]:
        return RunnableLambda(lambda _: self.route)


def node(route: Route) -> Any:
    return make_route_node(cast(BaseChatModel, StubLLM(route)))


async def run(route: Route) -> dict[str, Any]:
    return cast(dict[str, Any], await node(route)(cast(AgentState, {"question": "q"})))


async def test_research_sets_only_the_route() -> None:
    assert await run(Route(decision="research")) == {"route": "research"}


async def test_chat_fills_the_output_fields() -> None:
    out = await run(Route(decision="chat", reply=" Hello! "))
    assert out == {"route": "chat", "answer": "Hello!", "sources": [], "missing_info": None}


async def test_clarify_uses_reply_as_answer() -> None:
    out = await run(Route(decision="clarify", reply="Which one?"))
    assert out["route"] == "clarify" and out["answer"] == "Which one?"


async def test_missing_reply_falls_back_to_research() -> None:
    assert await run(Route(decision="clarify")) == {"route": "research"}
