from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from agent_smith.scraper.models import PageContent
from agent_smith.search.base import SearchResult


class AgentState(TypedDict):
    question: str
    queries: list[str]
    search_results: Annotated[list[SearchResult], add]
    pages: Annotated[list[PageContent], add]
    failed_urls: Annotated[list[str], add]
    iteration: int
    missing_info: str | None
    answer: str | None
    sources: list[str]

    messages: Annotated[list[AnyMessage], add_messages]
