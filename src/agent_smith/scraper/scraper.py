from types import TracebackType
from typing import Self

import httpx

from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import ParseError
from agent_smith.core.logging import get_logger
from agent_smith.scraper.extractor import extract_text
from agent_smith.scraper.fetcher import Fetcher
from agent_smith.scraper.models import PageContent
from agent_smith.scraper.parser import build_soup, parse_page

log = get_logger(__name__)


class Scraper:
    """Facade: fetch → parse → extract, producing an LLM-ready `PageContent`."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._fetcher = Fetcher(self._settings, client)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._fetcher.aclose()

    async def scrape(self, url: str) -> PageContent:
        result = await self._fetcher.fetch(url)
        soup = build_soup(result.text)
        page = parse_page(soup, result.final_url)  # before extract_text, which mutates soup
        text, truncated = extract_text(soup, self._settings.max_content_chars)
        if not text:
            raise ParseError(f"{url}: no readable text found")
        log.info("page_scraped", url=result.final_url, chars=len(text), truncated=truncated)
        return PageContent(
            url=page.url,
            title=page.title,
            description=page.description,
            language=page.language,
            text=text,
            links=page.links,
            truncated=truncated,
        )
