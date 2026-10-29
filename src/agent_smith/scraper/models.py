from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class FetchResult(_Model):
    """Raw outcome of downloading a page."""

    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str


class Link(_Model):
    url: str
    text: str = ""


class ParsedPage(_Model):
    """Metadata extracted from the HTML document."""

    url: str
    title: str = ""
    description: str = ""
    language: str = ""
    links: list[Link] = Field(default_factory=list)


class PageContent(_Model):
    """Final, LLM-ready representation of a scraped page."""

    url: HttpUrl
    title: str = ""
    description: str = ""
    language: str = ""
    text: str
    links: list[Link] = Field(default_factory=list)
    truncated: bool = False
