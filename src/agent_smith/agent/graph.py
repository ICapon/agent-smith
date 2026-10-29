from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_smith.agent.nodes.plan import make_plan_node
from agent_smith.agent.nodes.scrape import PageScraper, make_scrape_node
from agent_smith.agent.nodes.search import make_search_node
from agent_smith.agent.nodes.synthesize import make_synthesize_node
from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings, get_settings
from agent_smith.search.base import SearchProvider


class GraphInput(TypedDict):
    """What the caller provides."""

    question: str


class GraphOutput(TypedDict):
    """What the caller gets back; the rest of `AgentState` stays internal."""

    answer: str
    sources: list[str]
    missing_info: str | None


def make_route_after_synthesize(
    settings: Settings,
) -> Callable[[AgentState], Literal["plan", "__end__"]]:
    """Build the router that decides whether to loop back to `plan` or finish.

    Loops only while the answer still has gaps (`missing_info`) and iterations remain,
    so the cycle is always bounded by `Settings.max_iterations`.
    """

    def route_after_synthesize(state: AgentState) -> Literal["plan", "__end__"]:
        if state.get("missing_info") and state["iteration"] < settings.max_iterations:
            return "plan"
        return "__end__"  # the value of langgraph.graph.END, typed as a literal

    return route_after_synthesize


def build_graph(
    llm: BaseChatModel,
    scraper: PageScraper,
    provider: SearchProvider,
    settings: Settings | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    settings = settings or get_settings()

    builder = StateGraph(AgentState, input_schema=GraphInput, output_schema=GraphOutput)

    builder.add_node("plan", make_plan_node(llm, settings))
    builder.add_node("search", make_search_node(provider, settings))
    builder.add_node("scrape", make_scrape_node(scraper, settings))
    builder.add_node("synthesize", make_synthesize_node(llm))

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "scrape")
    builder.add_edge("scrape", "synthesize")
    builder.add_conditional_edges("synthesize", make_route_after_synthesize(settings))

    return builder.compile()
