from types import TracebackType
from typing import Self
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from bs4 import Tag

from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import SearchError
from agent_smith.core.logging import get_logger
from agent_smith.scraper.parser import build_soup
from agent_smith.search.base import SearchResult

log = get_logger(__name__)

_ENDPOINT = "https://html.duckduckgo.com/html/"
_BASE = "https://duckduckgo.com"


def _resolve(href: str) -> str | None:
    """Unwrap DuckDuckGo's `/l/?uddg=<url>` redirect links into the target URL."""
    url = urljoin(_BASE, href)
    parts = urlsplit(url)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg")
        if not target:
            return None
        url, parts = target[0], urlsplit(target[0])
    return url if parts.scheme in ("http", "https") and parts.netloc else None


def parse_results(html: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in build_soup(html).select("div.result"):
        if "result--ad" in (item.get("class") or []):
            continue
        link = item.select_one("a.result__a")
        href = link.get("href") if isinstance(link, Tag) else None
        if not isinstance(link, Tag) or not isinstance(href, str):
            continue
        url = _resolve(href)
        if url is None or url in seen:
            continue
        seen.add(url)
        snippet = item.select_one(".result__snippet")
        results.append(
            SearchResult(
                title=" ".join(link.get_text().split()),
                url=url,
                snippet=" ".join(snippet.get_text().split()) if snippet else "",
            )
        )
    return results


class DuckDuckGoSearch:
    """Search via DuckDuckGo's JavaScript-free HTML endpoint (no API key)."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": self._settings.user_agent},
            timeout=self._settings.http_timeout,
            follow_redirects=True,
        )

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
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise SearchError("empty query")
        limit = max_results or self._settings.max_search_results
        try:
            response = await self._client.post(_ENDPOINT, data={"q": query})
        except httpx.HTTPError as exc:
            raise SearchError(f"search request failed: {exc}") from exc
        if response.status_code != 200:
            # DuckDuckGo answers 202 (or 403/429) when it flags the client as a bot.
            raise SearchError(f"search blocked or failed (HTTP {response.status_code})")

        results = parse_results(response.text)
        if not results and "anomaly" in response.text:
            raise SearchError("search blocked by DuckDuckGo bot challenge")
        log.info("web_search", query=query, results=len(results))
        return results[:limit]
