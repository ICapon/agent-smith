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


class SandboxError(AgentSmithError):
    """Base class for sandbox failures."""


class SandboxUnavailableError(SandboxError):
    """The sandbox backend (e.g. the Docker CLI or daemon) cannot be used."""


class SandboxTimeoutError(SandboxError):
    """A sandboxed command exceeded its time limit and was killed."""

    def __init__(self, command: str, timeout: float) -> None:
        super().__init__(f"command timed out after {timeout:g}s: {command}")
        self.command = command
        self.timeout = timeout


class WorkspacePathError(SandboxError):
    """A path escapes the sandbox workspace or is otherwise not allowed."""


class AgentLimitError(AgentSmithError):
    """The agent used up its step budget before finishing."""
