import logging
from collections.abc import AsyncGenerator

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState

logger = logging.getLogger(__name__)

TOOL_NODE = None


def get_tool_node():
    global TOOL_NODE
    if TOOL_NODE is None:
        from agent.tools import TOOLS

        TOOL_NODE = ToolNode(TOOLS)
    return TOOL_NODE


def tools_node_wrapper(state: AgentState):
    """Wrapper for ToolNode - returns tool responses to be added to state."""
    logger.info(f"[TOOLS_WRAPPER] Input messages: {len(state['messages'])}")

    tool_node = get_tool_node()
    result = tool_node.invoke(state["messages"])

    sources = list(state.get("sources", []))

    for msg in result if isinstance(result, list) else [result]:
        if hasattr(msg, "name") and msg.name == "search_web":
            try:
                import re

                urls = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", msg.content)
                for title, url in urls:
                    sources.append({"title": title, "url": url})
            except Exception:
                pass

    tool_responses = result if isinstance(result, list) else [result]

    return {
        "messages": tool_responses,
        "sources": sources,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def create_agent_graph():
    """Create the LangGraph agent workflow with proper state management."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.nodes import build_system_prompt

    workflow = StateGraph(AgentState)

    def first_respond_node(state: AgentState):
        """First LLM call - returns new messages to be added."""
        import time

        logger.info("[FIRST_RESPOND] Initial LLM call")

        start_time = time.time()

        from agent.nodes import TOOLS, get_llm

        system_prompt = build_system_prompt()
        user_query = state["user_query"]

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]

        llm_with_tools = get_llm().bind_tools(TOOLS)
        response = llm_with_tools.invoke(messages)

        latency_ms = (time.time() - start_time) * 1000

        if hasattr(response, "tool_calls") and response.tool_calls:
            return {
                "messages": [response],
                "should_use_tools": True,
                "iteration_count": 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "total_latency_ms": latency_ms,
                },
            }

        return {
            "messages": [response],
            "should_use_tools": False,
            "final_response": response.content if hasattr(response, "content") else str(response),
            "metadata": {
                **state.get("metadata", {}),
                "total_latency_ms": latency_ms,
            },
        }

    def should_continue_tools(state: AgentState) -> str:
        """Determine whether to continue calling tools or generate final response."""
        messages = state["messages"]
        has_tool_calls = any(getattr(msg, "tool_calls", None) for msg in messages)

        if not has_tool_calls:
            return "respond"

        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= 6:
            logger.warning("Max iterations (6) reached")
            return "respond"

        return "continue"

    def respond_node(state: AgentState):
        """Generate the final response - returns new messages to be added."""
        import time

        from langchain_core.messages import SystemMessage

        from agent.nodes import build_system_prompt, get_llm

        logger.info("[RESPOND] Generating final response")

        start_time = time.time()

        system_prompt = build_system_prompt()
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state["messages"])

        llm = get_llm()
        response = llm.invoke(messages)

        response_text = response.content if hasattr(response, "content") else str(response)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "messages": [response],
            "final_response": response_text,
            "metadata": {
                **state.get("metadata", {}),
                "total_latency_ms": state.get("metadata", {}).get("total_latency_ms", 0)
                + latency_ms,
            },
        }

    workflow.add_node("first_respond", first_respond_node)
    workflow.add_node("tools", tools_node_wrapper)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("first_respond")

    workflow.add_conditional_edges(
        "first_respond",
        lambda state: "tools" if state.get("should_use_tools") else "respond",
        {"tools": "tools", "respond": "respond"},
    )

    workflow.add_conditional_edges(
        "tools", should_continue_tools, {"continue": "tools", "respond": "respond"}
    )

    workflow.add_edge("respond", END)

    compiled_graph = workflow.compile()

    logger.info("LangGraph agent compiled successfully")

    return compiled_graph


agent_graph = None


def get_agent_graph():
    global agent_graph
    if agent_graph is None:
        agent_graph = create_agent_graph()
    return agent_graph


def run_agent(chat_id: int, user_query: str, config: dict | None = None):
    """
    Run the agent with the given query.
    """
    global agent_graph
    agent_graph = None

    logger.info(f"[RUN_AGENT] Starting with query: {user_query[:50]}...")

    graph = get_agent_graph()

    initial_state = {
        "messages": [("user", user_query)],
        "user_query": user_query,
        "chat_id": chat_id,
        "tool_calls": [],
        "sources": [],
        "final_response": "",
        "metadata": {},
        "should_use_tools": False,
        "iteration_count": 0,
    }

    result = graph.invoke(initial_state)

    return {
        "final_response": result.get("final_response", "No response generated"),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    }


async def run_agent_streaming(chat_id: int, user_query: str) -> AsyncGenerator[dict, None]:
    """
    Run the agent with streaming support.
    """
    global agent_graph
    agent_graph = None

    logger.info(f"[RUN_AGENT_STREAMING] Starting with query: {user_query[:50]}...")

    graph = get_agent_graph()

    initial_state = {
        "messages": [("user", user_query)],
        "user_query": user_query,
        "chat_id": chat_id,
        "tool_calls": [],
        "sources": [],
        "final_response": "",
        "metadata": {},
        "should_use_tools": False,
        "iteration_count": 0,
    }

    accumulated_response = ""

    async for event in graph.astream(initial_state):
        for node_name, node_data in event.items():
            if node_name == "respond":
                if "final_response" in node_data:
                    yield {
                        "type": "respond",
                        "content": node_data["final_response"],
                        "sources": node_data.get("sources", []),
                        "metadata": node_data.get("metadata", {}),
                    }
                    accumulated_response = node_data["final_response"]
            elif node_name == "tools":
                yield {"type": "tools", "content": "Tool executed"}
            elif node_name == "first_respond" and "final_response" in node_data:
                yield {
                    "type": "respond",
                    "content": node_data["final_response"],
                    "sources": node_data.get("sources", []),
                    "metadata": node_data.get("metadata", {}),
                }
                accumulated_response = node_data["final_response"]

    yield {"type": "done", "content": accumulated_response}
