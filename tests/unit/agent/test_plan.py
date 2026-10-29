from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from agent_smith.agent.nodes.plan import QueryPlan, make_plan_node
from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings


class StubLLM:
    """Stands in for a chat model: with_structured_output returns a fixed plan."""

    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.seen: list[Any] = []

    def with_structured_output(self, schema: type[QueryPlan]) -> RunnableLambda[Any, QueryPlan]:
        def respond(prompt: Any) -> QueryPlan:
            self.seen.append(prompt)
            return schema(queries=self.queries)

        return RunnableLambda(respond)


def node(stub: StubLLM, **settings: Any) -> Any:
    return make_plan_node(cast(BaseChatModel, stub), Settings(_env_file=None, **settings))


async def test_plan_first_pass() -> None:
    stub = StubLLM(["a", "b"])
    state = cast(AgentState, {"question": "what?"})
    out = await node(stub)(state)
    assert out == {"queries": ["a", "b"], "iteration": 1}
    assert "none" in str(stub.seen[0].to_messages()[1].content)


async def test_plan_uses_missing_info_and_bumps_iteration() -> None:
    stub = StubLLM(["a"])
    state = cast(AgentState, {"question": "q", "iteration": 1, "missing_info": "the date"})
    out = await node(stub)(state)
    assert out["iteration"] == 2
    assert "the date" in str(stub.seen[0].to_messages()[1].content)


async def test_plan_dedupes_and_caps() -> None:
    stub = StubLLM([" a ", "a", "b", "c", "d"])
    out = await node(stub, max_queries=3)(cast(AgentState, {"question": "q"}))
    assert out["queries"] == ["a", "b", "c"]
