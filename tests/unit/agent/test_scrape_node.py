from typing import Any, cast

from agent_smith.agent.nodes.scrape import ScrapeNode, make_scrape_node
from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings
from agent_smith.core.errors import FetchError
from agent_smith.scraper.models import PageContent
from agent_smith.search.base import SearchResult


def hit(n: int) -> SearchResult:
    return SearchResult(title=f"T{n}", url=f"https://e.com/{n}")


class FakeScraper:
    """Scrapes n -> page n. `redirects` maps a URL to the final URL it lands on."""

    def __init__(self, fail: set[str] | None = None, redirects: dict[str, str] | None = None):
        self.fail = fail or set()
        self.redirects = redirects or {}
        self.calls: list[str] = []

    async def scrape(self, url: str) -> PageContent:
        self.calls.append(url)
        if url in self.fail:
            raise FetchError(url, "boom")
        return PageContent(url=self.redirects.get(url, url), title="t", text="x")


def node(scraper: FakeScraper, **kw: Any) -> ScrapeNode:
    return make_scrape_node(scraper, Settings(_env_file=None, **kw))


def state(**kw: Any) -> AgentState:
    return cast(AgentState, {"search_results": [hit(i) for i in range(1, 6)], **kw})


async def test_respects_cap_and_order() -> None:
    scraper = FakeScraper()
    out = await node(scraper, max_pages_per_iteration=2)(state())
    assert scraper.calls == ["https://e.com/1", "https://e.com/2"]
    assert out["scraped_urls"] == scraper.calls and out["failed_urls"] == []
    assert len(out["pages"]) == 2


async def test_records_failures_without_sinking_others() -> None:
    scraper = FakeScraper(fail={"https://e.com/1"})
    out = await node(scraper, max_pages_per_iteration=2)(state())
    assert out["failed_urls"] == ["https://e.com/1"]
    assert out["scraped_urls"] == ["https://e.com/2"]
    assert len(out["pages"]) == 1


async def test_skips_already_attempted_urls() -> None:
    scraper = FakeScraper()
    st = state(scraped_urls=["https://e.com/1"], failed_urls=["https://e.com/2"])
    await node(scraper, max_pages_per_iteration=2)(st)
    assert scraper.calls == ["https://e.com/3", "https://e.com/4"]


async def test_dedupes_pages_that_redirect_to_the_same_url() -> None:
    scraper = FakeScraper(redirects={"https://e.com/2": "https://e.com/1"})
    out = await node(scraper, max_pages_per_iteration=2)(state())
    assert len(out["pages"]) == 1
    assert out["scraped_urls"] == ["https://e.com/1", "https://e.com/2"]


async def test_nothing_to_do() -> None:
    out = await node(FakeScraper())(cast(AgentState, {}))
    assert out == {"pages": [], "scraped_urls": [], "failed_urls": []}
