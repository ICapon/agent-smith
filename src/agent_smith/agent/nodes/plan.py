from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_smith.agent.prompts import PLAN_PROMPT
from agent_smith.agent.state import AgentState
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.logging import get_logger

log = get_logger(__name__)

PlanNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


class QueryPlan(BaseModel):
    """Structured output the LLM must produce."""

    queries: list[str] = Field(
        min_length=1, description="Distinct search-engine queries, most important first."
    )


def make_plan_node(llm: BaseChatModel, settings: Settings | None = None) -> PlanNode:
    """Build the `plan` node, binding the LLM and settings it depends on."""
    settings = settings or get_settings()
    chain = PLAN_PROMPT | llm.with_structured_output(QueryPlan)

    async def plan(state: AgentState) -> dict[str, Any]:
        result = await chain.ainvoke(
            {
                "today": date.today().isoformat(),
                "max_queries": settings.max_queries,
                "question": state["question"],
                "missing_info": state.get("missing_info") or "none",
            }
        )
        # Defensive: the model may exceed max_queries or repeat itself.
        queries = list(dict.fromkeys(q.strip() for q in QueryPlan.model_validate(result).queries))
        queries = [q for q in queries if q][: settings.max_queries]

        iteration = state.get("iteration", 0) + 1
        log.info("plan", iteration=iteration, queries=queries)
        return {"queries": queries, "iteration": iteration}

    return plan
