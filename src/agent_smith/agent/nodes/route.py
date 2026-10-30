from datetime import date
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_smith.agent.prompts import ROUTE_PROMPT
from agent_smith.agent.state import AgentNode, AgentState
from agent_smith.core.logging import get_logger

log = get_logger(__name__)

RouteNode = AgentNode


class Route(BaseModel):
    """Structured output the LLM must produce."""

    decision: Literal["research", "chat", "clarify"] = Field(
        description="research: needs a web search; chat: answer directly; "
        "clarify: too ambiguous to act on."
    )
    reply: str | None = Field(
        default=None,
        description="For chat: the answer. For clarify: the question to ask the user. "
        "Null for research.",
    )


def make_route_node(llm: BaseChatModel) -> RouteNode:
    """Build the `route` node: decides whether a message needs research, a direct reply,
    or a clarifying question. Non-research decisions also fill the final output fields,
    so the graph can end right after this node."""
    chain = ROUTE_PROMPT | llm.with_structured_output(Route, method="json_schema")

    async def route(state: AgentState) -> dict[str, Any]:
        result = Route.model_validate(
            await chain.ainvoke({"today": date.today().isoformat(), "question": state["question"]})
        )
        reply = (result.reply or "").strip()
        # A direct reply without text is useless: fall back to researching.
        decision = result.decision if result.decision == "research" or reply else "research"

        log.info("route", decision=decision)
        if decision == "research":
            return {"route": decision}
        return {"route": decision, "answer": reply, "sources": [], "missing_info": None}

    return route
