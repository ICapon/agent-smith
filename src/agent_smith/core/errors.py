class AgentSmithError(Exception):
    """Base class for all errors raised by agent-smith."""


class ConfigError(AgentSmithError):
    """Invalid or missing configuration."""


class ScraperError(AgentSmithError):
    """Base class for scraper failures."""


class FetchError(ScraperError):
    """A page could not be downloaded."""

    def __init__(self, url: str, reason: str, *, status_code: int | None = None) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason
        self.status_code = status_code


class RobotsDisallowedError(ScraperError):
    """robots.txt forbids fetching the URL."""

    def __init__(self, url: str) -> None:
        super().__init__(f"{url}: disallowed by robots.txt")
        self.url = url


class ParseError(ScraperError):
    """The downloaded content could not be parsed or extracted."""


class SearchError(AgentSmithError):
    """A web search failed."""
