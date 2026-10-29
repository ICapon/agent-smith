import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from agent_smith.core.logging import get_logger

log = get_logger(__name__)

# Per-origin policy: a parsed robots.txt, or a blanket allow/deny.
_Policy = RobotFileParser | bool


class RobotsChecker:
    """Checks URLs against robots.txt, caching one policy per origin (RFC 9309).

    - robots.txt missing (4xx): everything is allowed.
    - robots.txt unreachable (5xx / network error): everything is disallowed.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self._client = client
        self._agent = user_agent.split("/")[0]
        self._policies: dict[str, _Policy] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        policy = self._policies.get(origin)
        if policy is None:
            async with self._locks.setdefault(origin, asyncio.Lock()):
                policy = self._policies.get(origin)
                if policy is None:
                    policy = await self._load(origin)
                    self._policies[origin] = policy
        if isinstance(policy, bool):
            return policy
        return policy.can_fetch(self._agent, url)

    async def _load(self, origin: str) -> _Policy:
        robots_url = f"{origin}/robots.txt"
        try:
            response = await self._client.get(robots_url)
        except httpx.HTTPError as exc:
            log.warning("robots_unreachable", url=robots_url, error=str(exc))
            return False
        if response.status_code >= 500:
            log.warning("robots_unavailable", url=robots_url, status=response.status_code)
            return False
        if response.status_code >= 400:
            return True
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser
