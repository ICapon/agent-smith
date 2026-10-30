import re
import shlex
from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Verdict(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    reason: str


# (command, reason) -> approved?  Whoever wires the agent decides how to ask the human.
Approver = Callable[[str, str], Awaitable[bool]]

# Read-only commands that are safe to run unattended when used on their own.
_ALLOWED_PROGRAMS = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "pwd", "echo", "diff", "sort", "uniq", "true"}
)
# Anything that chains, redirects, substitutes or expands makes a command more than its
# first word, so such commands never qualify for auto-allow.
_SHELL_META = frozenset(";&|<>`$(){}\n\\*?~")

_DENY_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), reason)
    for pattern, reason in (
        (r"\b(sudo|su|doas)\b", "privilege escalation"),
        (r"\brm\s+(-\S+\s+)*(/|/\*|~)(\s|$)", "recursive delete of / or home"),
        (r":\(\)\s*\{", "fork bomb"),
        (r"\bmkfs(\.\w+)?\b", "formatting a filesystem"),
        (r"\bdd\b[^|;&]*\bof=/dev/", "raw write to a device"),
        (r"\b(curl|wget)\b[^|;&]*\|\s*(sh|bash|zsh|python\d*)\b", "piping a download to a shell"),
        (r"docker\.sock", "access to the Docker socket"),
    )
)


class CommandPolicy:
    """Decides whether a shell command may run: allow, ask a human, or deny.

    This is a convenience layer, not a security boundary: string matching on shell
    commands is trivially bypassed. The sandbox is what actually contains a command; the
    policy keeps obviously hostile or destructive ones from being attempted and routes
    anything non-trivial to a human.
    """

    def __init__(self, default: Verdict = Verdict.ASK) -> None:
        if default is Verdict.DENY:
            raise ValueError("default must be ALLOW or ASK; use deny rules to block commands")
        self._default = default

    def evaluate(self, command: str) -> Decision:
        command = command.strip()
        if not command:
            return Decision(verdict=Verdict.DENY, reason="empty command")
        for pattern, reason in _DENY_RULES:
            if pattern.search(command):
                return Decision(verdict=Verdict.DENY, reason=reason)
        if self._is_simple_read(command):
            return Decision(verdict=Verdict.ALLOW, reason="read-only command")
        return Decision(verdict=self._default, reason="command is not on the allowlist")

    @staticmethod
    def _is_simple_read(command: str) -> bool:
        if _SHELL_META & set(command):
            return False
        try:
            words = shlex.split(command)
        except ValueError:
            return False
        return bool(words) and words[0] in _ALLOWED_PROGRAMS
