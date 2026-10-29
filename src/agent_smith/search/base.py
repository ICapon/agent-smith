from typing import Protocol

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    """Pluggable web-search backend."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...

    async def aclose(self) -> None: ...
