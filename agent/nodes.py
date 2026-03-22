import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

llm = None
streaming_llm = None


def get_llm():
    global llm
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        llm = ChatOpenAI(model="gpt-4o-mini")
    return llm


def get_streaming_llm():
    global streaming_llm
    if streaming_llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        streaming_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    return streaming_llm


def build_system_prompt() -> str:
    import os
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    tz_name = os.getenv("USER_TIMEZONE", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    now = datetime.now(tz)
    datetime_line = f"Current date and time: {now.strftime('%A, %B %d, %Y %H:%M %Z')}"

    location = os.getenv("USER_LOCATION", "")
    location_line = f"User's location: {location}" if location else ""

    prompt = (
        "You are a helpful personal assistant with web access. "
        "You are provided with the current date, time, and user location at the start of every message — use them when answering questions about time, scheduling, or location. "
        "Always use the search_web tool when the user asks about news, current events, prices, scores, or anything time-sensitive.\n\n"
        "Format all responses using Telegram MarkdownV2. Supported formatting:\n"
        "- *bold* for headings or key terms\n"
        "- _italic_ for emphasis\n"
        "- `code` for values, tickers, commands\n"
        "- [link text](url) for links\n"
        "For news or lists, use one item per line starting with a bullet (•). Keep responses concise and scannable on mobile.\n"
        "IMPORTANT: Escape special characters with backslash when they appear outside formatting: _ * [ ] ( ) ~ ` > # + - = | { } . !\n\n"
        f"{datetime_line}"
    )

    if location_line:
        prompt += f"\n{location_line}"

    return prompt


def should_continue_tools(state: AgentState) -> str:
    """Determine whether to continue calling tools or generate final response."""
    messages = state["messages"]

    has_tool_calls = any(  # type: ignore[union-attr]
        hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None) for msg in messages
    )

    if not has_tool_calls:
        return "respond"

    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= MAX_ITERATIONS:
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached")
        return "respond"

    return "continue"


def respond_node(state: AgentState) -> AgentState:
    """Generate the final response."""
    logger.info("[RESPOND] Generating final response")

    start_time = time.time()

    logger.info(f"[RESPOND] State messages count: {len(state['messages'])}")
    for i, msg in enumerate(state["messages"]):
        tc = getattr(msg, "tool_calls", None)
        tid = getattr(msg, "tool_call_id", None)
        logger.info(
            f"[RESPOND] Message {i}: {type(msg).__name__}, tool_calls={tc}, tool_call_id={tid}"
        )

    system_prompt = build_system_prompt()
    messages: list = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])  # type: ignore[arg-type]

    logger.info(f"[RESPOND] Final messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        tc = getattr(msg, "tool_calls", None)
        tid = getattr(msg, "tool_call_id", None)
        logger.info(
            f"[RESPOND] Final Message {i}: {type(msg).__name__}, tool_calls={tc}, tool_call_id={tid}"
        )

    llm = get_llm()
    response = llm.invoke(messages)

    response_text = response.content if hasattr(response, "content") else str(response)

    latency_ms = (time.time() - start_time) * 1000

    return {  # type: ignore
        **state,
        "messages": list(state["messages"]) + [AIMessage(content=response_text)],
        "final_response": response_text,
        "metadata": {
            **state.get("metadata", {}),
            "total_latency_ms": state.get("metadata", {}).get("total_latency_ms", 0) + latency_ms,
        },
    }


def first_respond_node(state: AgentState) -> AgentState:
    """First LLM call to determine if tools are needed."""
    logger.info("[FIRST_RESPOND] Initial LLM call")

    start_time = time.time()

    system_prompt = build_system_prompt()
    user_query = state["user_query"]

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]

    llm_with_tools = get_llm().bind_tools(TOOLS)
    response = llm_with_tools.invoke(messages)

    latency_ms = (time.time() - start_time) * 1000

    new_state = {
        **state,
        "messages": [HumanMessage(content=user_query), response],
        "metadata": {**state.get("metadata", {}), "total_latency_ms": latency_ms},
    }

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {**new_state, "should_use_tools": True, "iteration_count": 1}  # type: ignore[return-value]

    return {  # type: ignore[return-value]
        **new_state,
        "should_use_tools": False,
        "final_response": response.content if hasattr(response, "content") else str(response),
    }
