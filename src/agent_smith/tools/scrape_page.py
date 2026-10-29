from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_smith.core.errors import AgentSmithError
from agent_smith.scraper import Scraper


class ScrapePageInput(BaseModel):
    url: str = Field(description="Absolute http(s) URL of the page to read.")


def make_scrape_page_tool(scraper: Scraper) -> BaseTool:
    async def scrape_page(url: str) -> str:
        try:
            page = await scraper.scrape(url)
        except AgentSmithError as exc:
            return f"Could not read page: {exc}"
        header = [f"Title: {page.title}", f"URL: {page.url}"]
        if page.description:
            header.append(f"Description: {page.description}")
        if page.truncated:
            header.append("(content truncated)")
        return "\n".join(header) + "\n\n" + page.text

    return StructuredTool.from_function(
        coroutine=scrape_page,
        name="scrape_page",
        description="Fetch a web page and return its readable text content.",
        args_schema=ScrapePageInput,
    )
