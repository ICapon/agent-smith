from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from agent_smith.scraper.models import Link, ParsedPage


def build_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if isinstance(tag, Tag):
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _links(soup: BeautifulSoup, base_url: str) -> list[Link]:
    seen: set[str] = set()
    links: list[Link] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not isinstance(href, str):
            continue
        url, _ = urldefrag(urljoin(base_url, href.strip()))
        if urlsplit(url).scheme not in ("http", "https") or url in seen:
            continue
        seen.add(url)
        links.append(Link(url=url, text=" ".join(a.get_text().split())))
    return links


def parse_page(soup: BeautifulSoup, url: str) -> ParsedPage:
    """Extract metadata and absolute links. Does not modify `soup`."""
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    html = soup.find("html")
    lang = html.get("lang") if isinstance(html, Tag) else None
    return ParsedPage(
        url=url,
        title=title or _meta(soup, "og:title"),
        description=_meta(soup, "description", "og:description"),
        language=lang.strip() if isinstance(lang, str) else "",
        links=_links(soup, url),
    )
