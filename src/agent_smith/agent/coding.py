from datetime import date
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_smith.agent.prompts import CODING_PROMPT
from agent_smith.core.config import Settings, get_settings
from agent_smith.core.errors import AgentLimitError
from agent_smith.core.logging import get_logger

log = get_logger(__name__)


class CodingInput(TypedDict):
    """What the caller provides."""

    task: str


class CodingOutput(TypedDict):
    """What the caller gets back; the message history stays internal."""

    answer: str


class CodingState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str | None


def build_coding_agent(
    llm: BaseChatModel, tools: list[BaseTool], settings: Settings | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the tool-calling loop: `agent` decides, `tools` executes, repeat until done.

    The loop ends when the model replies without tool calls. Use `solve` to run it: it
    bounds the number of steps so a model that never stops cannot run forever.
    """
    settings = settings or get_settings()
    # One tool call at a time: approvals are interactive and commands share one workspace.
    chain = CODING_PROMPT | llm.bind_tools(tools, parallel_tool_calls=False)

    async def start(state: CodingState) -> dict[str, Any]:
        return {"messages": [HumanMessage(state["task"])]}

    async def agent(state: CodingState) -> dict[str, Any]:
        response = await chain.ainvoke(
            {"today": date.today().isoformat(), "messages": state["messages"]}
        )
        log.info("coding.agent", tool_calls=[call["name"] for call in response.tool_calls])
        answer = None if response.tool_calls else (response.text.strip() or "(no answer)")
        return {"messages": [response], "answer": answer}

    builder = StateGraph(CodingState, input_schema=CodingInput, output_schema=CodingOutput)
    builder.add_node("start", start)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    builder.add_edge(START, "start")
    builder.add_edge("start", "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")

    return builder.compile()


async def solve(
    agent: CompiledStateGraph[Any, Any, Any, Any], task: str, settings: Settings | None = None
) -> CodingOutput:
    """Run the coding agent on `task`, allowing at most `Settings.coding_max_steps` tool rounds."""
    settings = settings or get_settings()
    # start (1) + one agent and one tools superstep per round + the final agent reply (1).
    limit = 2 * settings.coding_max_steps + 2
    try:
        result = await agent.ainvoke({"task": task}, config={"recursion_limit": limit})
    except GraphRecursionError as exc:
        raise AgentLimitError(
            f"gave up after {settings.coding_max_steps} tool rounds without finishing"
        ) from exc
    return cast(CodingOutput, result)
