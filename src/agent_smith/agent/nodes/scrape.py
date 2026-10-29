import asyncio
from typing import Any, Protocol

from agent_smith.agent.state import AgentNode, AgentState
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import AgentSmithError
from agent_smith.core.logging import get_logger
from agent_smith.scraper.models import PageContent

log = get_logger(__name__)

ScrapeNode = AgentNode


class PageScraper(Protocol):
    async def scrape(self, url: str) -> PageContent: ...


def make_scrape_node(scraper: PageScraper, settings: Settings | None = None) -> ScrapeNode:
    """Build the `scrape` node: reads the best not-yet-visited search results."""
    settings = settings or get_settings()

    async def read(url: str) -> PageContent | None:
        try:
            return await scraper.scrape(url)
        except AgentSmithError as exc:
            log.warning("scrape_failed", url=url, error=str(exc))
            return None

    async def scrape(state: AgentState) -> dict[str, Any]:
        # `scraped_urls` holds the URLs we *requested*: a redirect changes `page.url`,
        # so comparing against pages alone would re-fetch the same result next iteration.
        attempted = set(state.get("scraped_urls", [])) | set(state.get("failed_urls", []))
        todo = [r.url for r in state.get("search_results", []) if r.url not in attempted]
        todo = todo[: settings.max_pages_per_iteration]

        outcomes = await asyncio.gather(*(read(url) for url in todo))

        seen_pages = {str(p.url) for p in state.get("pages", [])}
        pages: list[PageContent] = []
        scraped: list[str] = []
        failed: list[str] = []
        for url, page in zip(todo, outcomes, strict=True):
            if page is None:
                failed.append(url)
                continue
            scraped.append(url)
            if str(page.url) not in seen_pages:  # two results may redirect to one page
                seen_pages.add(str(page.url))
                pages.append(page)

        log.info("scrape", requested=len(todo), pages=len(pages), failed=len(failed))
        return {"pages": pages, "scraped_urls": scraped, "failed_urls": failed}

    return scrape
