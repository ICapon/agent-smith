import asyncio
from types import TracebackType
from typing import Self
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import FetchError, RobotsDisallowedError
from agent_smith.core.logging import get_logger
from agent_smith.scraper.models import FetchResult
from agent_smith.scraper.robots import RobotsChecker

log = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_HTML_TYPES = ("text/html", "application/xhtml+xml")


class _Transient(Exception):
    """Internal marker for failures worth retrying."""

    def __init__(self, reason: str, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


class Fetcher:
    """Async HTML downloader: robots-aware, bounded concurrency, retries, size cap."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": self._settings.user_agent, "Accept": ", ".join(_HTML_TYPES)},
            timeout=self._settings.http_timeout,
            follow_redirects=True,
            max_redirects=5,
        )
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)
        self._robots = (
            RobotsChecker(self._client, self._settings.user_agent)
            if self._settings.respect_robots
            else None
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

    async def fetch(self, url: str) -> FetchResult:
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
            raise FetchError(url, "only absolute http(s) URLs are supported")
        if self._robots is not None and not await self._robots.allowed(url):
            raise RobotsDisallowedError(url)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._settings.http_max_retries + 1),
            wait=wait_exponential_jitter(multiplier=0.5, max=8.0),
            retry=retry_if_exception_type(_Transient),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    async with self._semaphore:
                        return await self._fetch_once(url)
        except _Transient as exc:
            raise FetchError(url, exc.reason, status_code=exc.status_code) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def _fetch_once(self, url: str) -> FetchResult:
        try:
            async with self._client.stream("GET", url) as response:
                status = response.status_code
                if status in _RETRYABLE_STATUS:
                    raise _Transient(f"HTTP {status}", status)
                if status >= 400:
                    raise FetchError(url, f"HTTP {status}", status_code=status)

                content_type = response.headers.get("content-type", "")
                if not content_type.lower().startswith(_HTML_TYPES):
                    raise FetchError(url, f"unsupported content type {content_type!r}")

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._settings.max_response_bytes:
                        raise FetchError(url, "response exceeds max_response_bytes")

                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=status,
                    content_type=content_type,
                    text=bytes(body).decode(response.encoding or "utf-8", errors="replace"),
                )
        except httpx.TimeoutException as exc:
            raise _Transient("timeout") from exc
        except httpx.TransportError as exc:
            raise _Transient(f"network error: {exc}") from exc
