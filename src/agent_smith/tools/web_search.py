from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_smith.core.errors import SearchError
from agent_smith.search.base import SearchProvider


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query, as you would type it in a search engine.")


def make_web_search_tool(provider: SearchProvider, max_results: int) -> BaseTool:
    async def web_search(query: str) -> str:
        try:
            results = await provider.search(query, max_results)
        except SearchError as exc:
            return f"Search failed: {exc}"
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)
        )

    return StructuredTool.from_function(
        coroutine=web_search,
        name="web_search",
        description=(
            "Search the web. Returns a numbered list of results with title, URL and snippet. "
            "Use scrape_page on a URL to read the full content."
        ),
        args_schema=WebSearchInput,
    )
