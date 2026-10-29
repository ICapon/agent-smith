from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent_smith.agent.prompts import SYNTHESIZE_PROMPT, format_sources
from agent_smith.agent.state import AgentNode, AgentState
from agent_smith.core.logging import get_logger

log = get_logger(__name__)

SynthesizeNode = AgentNode


class Synthesis(BaseModel):
    """Structured output the LLM must produce."""

    answer: str = Field(description="The answer, citing sources inline as [1], [2].")
    cited_sources: list[int] = Field(
        default_factory=list, description="Ids of the sources actually used in the answer."
    )
    missing_info: str | None = Field(
        default=None,
        description="What is still missing to answer fully, or null if the answer is complete.",
    )


def make_synthesize_node(llm: BaseChatModel) -> SynthesizeNode:
    """Build the `synthesize` node: answers from the gathered pages and flags any gap."""
    chain = SYNTHESIZE_PROMPT | llm.with_structured_output(Synthesis, method="json_schema")

    async def synthesize(state: AgentState) -> dict[str, Any]:
        pages = state.get("pages", [])
        if not pages:
            # Nothing to ground an answer on: don't ask the LLM to improvise.
            log.warning("synthesize_no_pages")
            return {
                "answer": "I could not read any sources to answer this question.",
                "sources": [],
                "missing_info": "No sources could be retrieved; try different search queries.",
            }

        result = Synthesis.model_validate(
            await chain.ainvoke(
                {
                    "today": date.today().isoformat(),
                    "question": state["question"],
                    "sources": format_sources(pages),
                }
            )
        )

        # Citation ids are 1-based positions in `pages`; drop any the model invented.
        cited = sorted({i for i in result.cited_sources if 1 <= i <= len(pages)})
        missing = (result.missing_info or "").strip() or None

        log.info("synthesize", cited=cited, complete=missing is None)
        return {
            "answer": result.answer.strip(),
            "sources": [str(pages[i - 1].url) for i in cited],
            "missing_info": missing,
        }

    return synthesize
