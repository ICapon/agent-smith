import re

from bs4 import BeautifulSoup, Tag

_NOISE_TAGS = (
    "script", "style", "noscript", "template", "svg", "iframe", "canvas",
    "nav", "header", "footer", "aside", "form", "button", "dialog",
)  # fmt: skip
_BLANK_LINES = re.compile(r"\n{3,}")


def _main_container(soup: BeautifulSoup) -> Tag:
    for candidate in (
        soup.find("main"),
        soup.find("article"),
        soup.select_one("[role=main]"),
        soup.body,
    ):
        if isinstance(candidate, Tag):
            return candidate
    return soup


def extract_text(soup: BeautifulSoup, max_chars: int) -> tuple[str, bool]:
    """Return the page's readable text and whether it was truncated.

    Destructive: strips noise elements from `soup`, so parse links/metadata first.
    """
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    raw = _main_container(soup).get_text("\n")
    lines = (" ".join(line.split()) for line in raw.splitlines())
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    boundary = cut.rfind("\n")
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return cut.rstrip(), True
