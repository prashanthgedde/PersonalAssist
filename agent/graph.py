import logging
from collections.abc import AsyncGenerator

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState

logger = logging.getLogger(__name__)

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
    from agent.nodes import first_respond_node, respond_node, should_continue_tools  # noqa: F401

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
    return agent_graph  # type: ignore[return-value]


def run_agent(chat_id: int, user_query: str, config: dict | None = None):
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

    final_state = None
    for state in graph.stream(initial_state):  # type: ignore[arg-type]
        final_state = state
        logger.debug(f"Graph state: {list(state.keys())}")

    if final_state:
        for node_data in final_state.values():
            if "final_response" in node_data:
                return node_data

    return {"final_response": "No response generated", "sources": [], "metadata": {}}


async def run_agent_streaming(chat_id: int, user_query: str) -> AsyncGenerator[dict, None]:
    """
    Run the agent with streaming support.

    Yields:
        dict with keys: 'type' (stream/respond/tools/done), 'content' (text), etc.
    """
    global agent_graph
    agent_graph = None

    logger.info(f"[RUN_AGENT_STREAMING] Starting with query: {user_query[:50]}...")

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

    accumulated_response = ""

    logger.info("[LANGGRAPH_STREAMING] Starting streaming graph execution")

    async for event in graph.astream_events(initial_state, version="v2"):
        event_type = event.get("event")

        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                if isinstance(chunk.content, list):
                    for part in chunk.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            accumulated_response += part.get("text", "")
                        elif isinstance(part, str):
                            accumulated_response += part
                elif isinstance(chunk.content, str):
                    accumulated_response += chunk.content

                yield {
                    "type": "stream",
                    "chunk": chunk.content,
                    "content": accumulated_response,
                }

        elif event_type == "on_chain_end":
            node_name = event.get("name", "")
            if node_name == "tools":
                yield {"type": "tools", "content": "Tools executed"}
            elif node_name == "respond":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_response = output.get("final_response", accumulated_response)
                    sources = output.get("sources", [])
                    metadata = output.get("metadata", {})
                    yield {
                        "type": "respond",
                        "content": final_response,
                        "sources": sources,
                        "metadata": metadata,
                    }

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            yield {"type": "tools_start", "content": f"Calling {tool_name}..."}

        elif event_type == "on_tool_end":
            yield {"type": "tools_end", "content": "Tool complete"}

    yield {"type": "done", "content": accumulated_response}
