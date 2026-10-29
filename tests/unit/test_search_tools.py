import httpx
import pytest
import respx

from agent_smith.core.config import Settings
from agent_smith.core.errors import SearchError
from agent_smith.scraper import Scraper
from agent_smith.search import DuckDuckGoSearch
from agent_smith.search.duckduckgo import parse_results
from agent_smith.tools import build_tools

DDG_HTML = """<html><body>
<div class="result result--ad"><a class="result__a" href="https://ad.com">Ad</a></div>
<div class="result"><h2><a class="result__a"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=1">Example  A</a></h2>
 <a class="result__snippet">Snippet <b>A</b></a></div>
<div class="result"><a class="result__a" href="https://example.org/b">B</a></div>
<div class="result"><a class="result__a" href="https://example.org/b">B dup</a></div>
<div class="result"><a class="result__a" href="/l/?kh=-1">no target</a></div>
</body></html>"""

ENDPOINT = "https://html.duckduckgo.com/html/"


def settings() -> Settings:
    return Settings(_env_file=None, respect_robots=False, http_max_retries=0)


def test_parse_results() -> None:
    results = parse_results(DDG_HTML)
    assert [(r.title, r.url, r.snippet) for r in results] == [
        ("Example A", "https://example.com/a?x=1", "Snippet A"),
        ("B", "https://example.org/b", ""),
    ]


@respx.mock
async def test_search_limits_results() -> None:
    route = respx.post(ENDPOINT).respond(200, html=DDG_HTML)
    async with DuckDuckGoSearch(settings()) as ddg:
        results = await ddg.search("  python ", max_results=1)
    assert len(results) == 1
    assert b"q=python" in route.calls[0].request.content


@respx.mock
@pytest.mark.parametrize(
    "response",
    [httpx.Response(202, html="x"), httpx.Response(200, html="<div class='anomaly-modal'>")],
)
async def test_search_blocked(response: httpx.Response) -> None:
    respx.post(ENDPOINT).mock(return_value=response)
    async with DuckDuckGoSearch(settings()) as ddg:
        with pytest.raises(SearchError):
            await ddg.search("q")


async def test_search_empty_query() -> None:
    async with DuckDuckGoSearch(settings()) as ddg:
        with pytest.raises(SearchError):
            await ddg.search("  ")


@respx.mock
async def test_tools_end_to_end() -> None:
    respx.post(ENDPOINT).respond(200, html=DDG_HTML)
    respx.get("https://example.com/a").respond(
        200, html="<html><title>T</title><body><main>Hello</main></body></html>"
    )
    respx.get("https://example.org/b").respond(404)
    async with DuckDuckGoSearch(settings()) as ddg, Scraper(settings()) as scraper:
        search, scrape = build_tools(scraper, ddg, max_results=5)
        assert search.name == "web_search" and scrape.name == "scrape_page"

        listing = await search.ainvoke({"query": "python"})
        assert "1. Example A" in listing and "https://example.org/b" in listing

        page = await scrape.ainvoke({"url": "https://example.com/a"})
        assert "Title: T" in page and page.endswith("Hello")

        failure = await scrape.ainvoke({"url": "https://example.org/b"})
        assert failure.startswith("Could not read page:")


@respx.mock
async def test_search_tool_reports_errors_to_the_llm() -> None:
    respx.post(ENDPOINT).respond(202)
    async with DuckDuckGoSearch(settings()) as ddg, Scraper(settings()) as scraper:
        search, _ = build_tools(scraper, ddg, max_results=5)
        assert (await search.ainvoke({"query": "x"})).startswith("Search failed:")
