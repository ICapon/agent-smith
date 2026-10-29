from langchain_core.tools import BaseTool

from agent_smith.scraper import Scraper
from agent_smith.search.base import SearchProvider
from agent_smith.tools.scrape_page import make_scrape_page_tool
from agent_smith.tools.web_search import make_web_search_tool


def build_tools(scraper: Scraper, provider: SearchProvider, max_results: int) -> list[BaseTool]:
    """Build the agent's tools. The caller owns (and closes) `scraper` and `provider`."""
    return [make_web_search_tool(provider, max_results), make_scrape_page_tool(scraper)]


__all__ = ["build_tools"]
