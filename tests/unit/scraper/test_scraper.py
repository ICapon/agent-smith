import httpx
import pytest
import respx

from agent_smith.core.config import Settings
from agent_smith.core.errors import FetchError, ParseError, RobotsDisallowedError
from agent_smith.scraper import Scraper
from agent_smith.scraper.extractor import extract_text
from agent_smith.scraper.fetcher import Fetcher
from agent_smith.scraper.parser import build_soup, parse_page

HTML = """<html lang="es"><head><title> Hola  mundo </title>
<meta name="description" content="Una página"></head>
<body><nav>menu</nav><main><h1>Título</h1><p>Contenido   útil.</p>
<script>evil()</script><a href="/a#x">A</a><a href="/a">dup</a>
<a href="mailto:x@y.z">mail</a></main><footer>pie</footer></body></html>"""


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, http_max_retries=2, **kw)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None: ...

    monkeypatch.setattr("asyncio.sleep", instant)


def test_parse_and_extract() -> None:
    soup = build_soup(HTML)
    page = parse_page(soup, "https://e.com/x")
    assert (page.title, page.description, page.language) == ("Hola mundo", "Una página", "es")
    assert [link.url for link in page.links] == ["https://e.com/a"]
    text, truncated = extract_text(soup, 1000)
    assert text.splitlines()[:2] == ["Título", "Contenido útil."]
    assert "evil" not in text and "menu" not in text and "pie" not in text
    assert not truncated


def test_extract_truncates_on_line_boundary() -> None:
    soup = build_soup("<body>" + "".join(f"<p>line{i}</p>" for i in range(100)) + "</body>")
    text, truncated = extract_text(soup, 50)
    assert truncated and len(text) <= 50 and text.endswith(tuple("0123456789"))


@respx.mock
async def test_scrape_end_to_end() -> None:
    respx.get("https://e.com/robots.txt").respond(404)
    respx.get("https://e.com/p").respond(200, html=HTML)
    async with Scraper(settings()) as scraper:
        page = await scraper.scrape("https://e.com/p")
    assert page.title == "Hola mundo"
    assert "evil" not in page.text


@respx.mock
async def test_robots_disallow() -> None:
    respx.get("https://e.com/robots.txt").respond(200, text="User-agent: *\nDisallow: /private")
    respx.get("https://e.com/ok").respond(200, html=HTML)
    async with Fetcher(settings()) as fetcher:
        with pytest.raises(RobotsDisallowedError):
            await fetcher.fetch("https://e.com/private/x")
        assert (await fetcher.fetch("https://e.com/ok")).status_code == 200


@respx.mock
async def test_robots_unreachable_disallows() -> None:
    respx.get("https://e.com/robots.txt").respond(503)
    async with Fetcher(settings()) as fetcher:
        with pytest.raises(RobotsDisallowedError):
            await fetcher.fetch("https://e.com/")


@respx.mock
async def test_retries_then_succeeds() -> None:
    route = respx.get("https://e.com/p").mock(
        side_effect=[httpx.Response(503), httpx.ConnectError("x"), httpx.Response(200, html=HTML)]
    )
    async with Fetcher(settings(respect_robots=False)) as fetcher:
        assert (await fetcher.fetch("https://e.com/p")).status_code == 200
    assert route.call_count == 3


@respx.mock
async def test_retries_exhausted() -> None:
    route = respx.get("https://e.com/p").respond(503)
    async with Fetcher(settings(respect_robots=False)) as fetcher:
        with pytest.raises(FetchError) as info:
            await fetcher.fetch("https://e.com/p")
    assert info.value.status_code == 503 and route.call_count == 3


@respx.mock
@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(404, html="x"), "HTTP 404"),
        (httpx.Response(200, json={"a": 1}), "content type"),
        (httpx.Response(200, html="x" * 100), "max_response_bytes"),
    ],
)
async def test_fetch_errors_not_retried(response: httpx.Response, match: str) -> None:
    route = respx.get("https://e.com/p").mock(return_value=response)
    async with Fetcher(settings(respect_robots=False, max_response_bytes=50)) as fetcher:
        with pytest.raises(FetchError, match=match):
            await fetcher.fetch("https://e.com/p")
    assert route.call_count == 1


async def test_rejects_non_http_url() -> None:
    async with Fetcher(settings(respect_robots=False)) as fetcher:
        with pytest.raises(FetchError):
            await fetcher.fetch("file:///etc/passwd")


@respx.mock
async def test_empty_page_raises_parse_error() -> None:
    respx.get("https://e.com/p").respond(200, html="<html><body><script>x</script></body></html>")
    async with Scraper(settings(respect_robots=False)) as scraper:
        with pytest.raises(ParseError):
            await scraper.scrape("https://e.com/p")
