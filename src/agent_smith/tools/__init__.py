from langchain_core.tools import BaseTool

from agent_smith.core.config import Settings, get_settings
from agent_smith.sandbox import Approver, CommandPolicy, Sandbox
from agent_smith.scraper import Scraper
from agent_smith.search.base import SearchProvider
from agent_smith.tools.run_command import make_run_command_tool
from agent_smith.tools.scrape_page import make_scrape_page_tool
from agent_smith.tools.web_search import make_web_search_tool
from agent_smith.tools.write_file import make_write_file_tool


def build_tools(scraper: Scraper, provider: SearchProvider, max_results: int) -> list[BaseTool]:
    """Build the agent's tools. The caller owns (and closes) `scraper` and `provider`."""
    return [make_web_search_tool(provider, max_results), make_scrape_page_tool(scraper)]


def build_sandbox_tools(
    sandbox: Sandbox,
    policy: CommandPolicy | None = None,
    approver: Approver | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """Build the code tools. The caller owns (and closes) `sandbox`.

    Without an `approver`, any command the policy does not allow outright is refused.
    """
    settings = settings or get_settings()
    return [
        make_run_command_tool(sandbox, policy or CommandPolicy(), approver),
        make_write_file_tool(sandbox, settings.sandbox_max_file_bytes),
    ]


__all__ = ["build_sandbox_tools", "build_tools"]
