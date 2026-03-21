import logging
import time
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.tools import TOOLS, TOOL_MAP

logger = logging.getLogger(__name__)

llm = None


def get_llm():
    global llm
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        llm = ChatOpenAI(model="gpt-4o-mini")
    return llm


def build_system_prompt() -> str:
    from datetime import datetime, timezone
    import os
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
        "For news or lists, use one item per line starting with a bullet (\u2022). Keep responses concise and scannable on mobile.\n"
        "IMPORTANT: Escape special characters with backslash when they appear outside formatting: _ * [ ] ( ) ~ ` > # + - = | { } . !\n\n"
        f"{datetime_line}"
    )

    if location_line:
        prompt += f"\n{location_line}"

    return prompt


def run_agent(chat_id: int, user_query: str, config: dict = None):
    """
    Simple sequential agent (SINGLE ROUND):
    1. LLM call -> decides to use tools?
    2. Execute ALL tools in parallel (one shot)
    3. LLM call with results -> final response

    Limitation: Cannot call more tools after seeing results.
    """
    logger.info(f"[AGENT] Processing query: {user_query[:50]}...")

    start_time = time.time()

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    system_msg = SystemMessage(content=build_system_prompt())
    messages = [system_msg, HumanMessage(content=user_query)]

    logger.info("[AGENT] First LLM call")
    response = llm_with_tools.invoke(messages)

    tool_calls = []
    sources = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(
            f"[AGENT] Tool calls needed: {[tc.get('name') for tc in response.tool_calls]}"
        )

        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")

            logger.info(f"[AGENT] Calling tool: {tool_name} with args: {tool_args}")

            tool = TOOL_MAP.get(tool_name)
            if tool:
                result = tool.invoke(tool_args)
                logger.info(f"[AGENT] Tool result: {str(result)[:100]}...")

                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": str(result),
                    }
                )

                if tool_name == "search_web":
                    urls = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", str(result))
                    for title, url in urls:
                        sources.append({"title": title, "url": url})

                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id or "default")
                )

        messages.append(response)
        messages.extend(tool_messages)

        logger.info("[AGENT] Second LLM call with tool results")
        final_response = llm.invoke(messages)
        logger.info(f"[AGENT] Final response: {final_response}")
        final_text = ""
        if hasattr(final_response, "content"):
            final_text = final_response.content or ""
        elif hasattr(final_response, "text"):
            final_text = final_response.text or ""
        else:
            final_text = str(final_response)
    else:
        logger.info("[AGENT] No tools needed")
        logger.info(f"[AGENT] Response: {response}")
        final_text = ""
        if hasattr(response, "content"):
            final_text = response.content or ""
        elif hasattr(response, "text"):
            final_text = response.text or ""
        else:
            final_text = str(response)

    total_latency_ms = (time.time() - start_time) * 1000

    return {
        "final_response": final_text,
        "sources": sources,
        "metadata": {
            "tool_calls": len(tool_calls),
            "total_latency_ms": total_latency_ms,
            "iteration_count": 1 if tool_calls else 0,
        },
    }
