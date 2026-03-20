import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState

logger = logging.getLogger(__name__)

TOOL_NODE = None


def get_tool_node():
    global TOOL_NODE
    if TOOL_NODE is None:
        from agent.tools import TOOLS

        TOOL_NODE = ToolNode(TOOLS)
    return TOOL_NODE


def tools_node_wrapper(state):
    """Wrapper for ToolNode to handle state properly."""
    logger.info(f"[TOOLS_WRAPPER] Input messages: {len(state['messages'])}")
    for i, msg in enumerate(state["messages"]):
        logger.info(
            f"[TOOLS_WRAPPER] Message {i}: {type(msg).__name__}, tool_calls: {getattr(msg, 'tool_calls', None)}"
        )

    tool_node = get_tool_node()
    result = tool_node.invoke(state["messages"])

    logger.info(f"[TOOLS_WRAPPER] ToolNode result: {result}")
    logger.info(f"[TOOLS_WRAPPER] Result type: {type(result)}")

    new_messages = list(state["messages"])
    if isinstance(result, list):
        new_messages.extend(result)
    else:
        new_messages.append(result)

    logger.info(f"[TOOLS_WRAPPER] New messages count: {len(new_messages)}")

    tool_calls_count = 0
    sources = list(state.get("sources", []))

    for msg in result if isinstance(result, list) else [result]:
        logger.info(
            f"[TOOLS_WRAPPER] Result msg: {type(msg).__name__}, name: {getattr(msg, 'name', None)}, tool_call_id: {getattr(msg, 'tool_call_id', None)}"
        )
        if hasattr(msg, "name") and msg.name == "search_web":
            try:
                import re

                urls = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", msg.content)
                for title, url in urls:
                    sources.append({"title": title, "url": url})
            except Exception:
                pass

    for msg in new_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)

    return {
        "messages": new_messages,
        "sources": sources,
        "tool_calls": tool_calls_count,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "metadata": {
            **state.get("metadata", {}),
            "tool_calls": tool_calls_count,
            "iteration_count": state.get("iteration_count", 0) + 1,
        },
    }


def create_agent_graph():
    """Create the LangGraph agent workflow."""
    from agent.nodes import should_continue_tools, first_respond_node, respond_node

    workflow = StateGraph(AgentState)

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


def run_agent(chat_id: int, user_query: str, config: dict = None):
    """
    Run the agent with the given query.

    Args:
        chat_id: Telegram chat ID
        user_query: User's message
        config: Optional config dict with thread_id

    Returns:
        Final agent response
    """
    global agent_graph
    agent_graph = None

    logger.info(f"[RUN_AGENT] Starting with query: {user_query[:50]}...")

    graph = get_agent_graph()

    initial_state = {
        "messages": [],
        "user_query": user_query,
        "chat_id": chat_id,
        "tool_calls": [],
        "sources": [],
        "final_response": "",
        "metadata": {},
        "should_use_tools": False,
        "iteration_count": 0,
    }
    logger.info(f"[RUN_AGENT] Initial state: {initial_state}")

    checkpoint_config = None

    final_state = None
    if checkpoint_config:
        for state in graph.stream(initial_state, checkpoint_config):
            final_state = state
            logger.debug(f"Graph state: {list(state.keys())}")
    else:
        for state in graph.stream(initial_state):
            final_state = state
            logger.debug(f"Graph state: {list(state.keys())}")

    if final_state:
        for node_data in final_state.values():
            if "final_response" in node_data:
                return node_data

    return {"final_response": "No response generated", "sources": [], "metadata": {}}
