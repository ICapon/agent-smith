from operator import add
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from agent_smith.scraper.models import PageContent
from agent_smith.search.base import SearchResult


class AgentState(TypedDict):
    question: str
    route: str | None
    queries: list[str]
    search_results: Annotated[list[SearchResult], add]
    pages: Annotated[list[PageContent], add]
    scraped_urls: Annotated[list[str], add]
    failed_urls: Annotated[list[str], add]
    iteration: int
    missing_info: str | None
    answer: str | None
    sources: list[str]

    messages: Annotated[list[AnyMessage], add_messages]


class AgentNode(Protocol):
    """An async graph node: reads the state, returns only the fields it changes.

    A Protocol (not a bare Callable) because LangGraph's typing matches the `state` name.
    """

    async def __call__(self, state: AgentState) -> dict[str, Any]: ...
