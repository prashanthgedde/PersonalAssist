import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.tools import TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

llm = None


def get_llm():
    global llm
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        llm = ChatOpenAI(model="gpt-4o-mini")
    return llm


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
        "For news or lists, use one item per line starting with a bullet (\u2022). Keep responses concise and scannable on mobile.\n"
        "IMPORTANT: Escape special characters with backslash when they appear outside formatting: _ * [ ] ( ) ~ ` > # + - = | { } . !\n\n"
        f"{datetime_line}"
    )

    if location_line:
        prompt += f"\n{location_line}"

    return prompt


def has_tool_calls(response) -> bool:
    """Check if response contains tool calls."""
    if hasattr(response, "tool_calls") and response.tool_calls:
        return len(response.tool_calls) > 0
    return False


def extract_sources_from_result(result_content: str):
    """Extract URLs from search results."""
    sources = []
    urls = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", result_content)
    for title, url in urls:
        sources.append({"title": title, "url": url})
    return sources


def run_agent(chat_id: int, user_query: str, config: dict | None = None):
    """
    Multi-turn agentic loop (ITERATIVE):

    Loop structure:
    ┌────────────────────────────────────────────────────┐
    │ while iteration < MAX_ITERATIONS (6):              │
    │                                                    │
    │   1. LLM call with tools                           │
    │          ↓                                         │
    │   2. No tool_calls? → break, return response       │
    │          ↓                                         │
    │   3. Execute tools, accumulate messages            │
    │          ↓                                         │
    │   4. Loop back to step 1                           │
    │                                                    │
    │ Example: "Stock market news → how does it affect   │
    │ Apple?"                                            │
    │ → Iteration 1: search_web (market news)             │
    │ → Iteration 2: get_stock (AAPL)                    │
    │ → Iteration 3: final response                      │
    └────────────────────────────────────────────────────┘

    Key difference from simple_agent:
    - Can call multiple ROUNDS of tools
    - Each round, LLM decides if more tools are needed
    - Stops at MAX_ITERATIONS to prevent infinite loops
    """
    logger.info(f"[AGENT] Processing query: {user_query[:50]}...")

    start_time = time.time()

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    system_msg = SystemMessage(content=build_system_prompt())
    messages = [system_msg, HumanMessage(content=user_query)]

    tool_calls = []
    sources = []
    iteration_count = 0
    final_response = None

    while iteration_count < MAX_ITERATIONS:
        iteration_count += 1
        logger.info(f"[AGENT] Iteration {iteration_count}: LLM call")

        response = llm_with_tools.invoke(messages)

        if not has_tool_calls(response):
            logger.info(
                f"[AGENT] Iteration {iteration_count}: No tool calls, generating final response"
            )
            messages.append(response)
            final_response = response
            break

        logger.info(
            f"[AGENT] Iteration {iteration_count}: Tool calls: {[tc.get('name') for tc in response.tool_calls]}"
        )

        messages.append(response)

        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")

            logger.info(
                f"[AGENT] Iteration {iteration_count}: Calling {tool_name} with {tool_args}"
            )

            tool = TOOL_MAP.get(tool_name)
            if tool:
                result = tool.invoke(tool_args)
                result_str = str(result)
                logger.info(f"[AGENT] Tool result: {result_str[:100]}...")

                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": result_str,
                    }
                )

                if tool_name == "search_web":
                    sources.extend(extract_sources_from_result(result_str))

                tool_messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id or f"call_{iteration_count}",
                    )
                )

        messages.extend(tool_messages)
        logger.info(f"[AGENT] Iteration {iteration_count}: Tool execution complete")

    final_text = ""
    if final_response is None:
        final_text = "Agent did not produce a response"
    else:
        response_obj = final_response
        final_text = (
            getattr(response_obj, "content", None)
            or getattr(response_obj, "text", None)
            or str(response_obj)
        )

    total_latency_ms = (time.time() - start_time) * 1000

    logger.info(
        f"[AGENT] Done. iterations={iteration_count}, tools={len(tool_calls)}, latency={total_latency_ms:.0f}ms"
    )

    return {
        "final_response": final_text,
        "sources": sources,
        "metadata": {
            "tool_calls": len(tool_calls),
            "total_latency_ms": total_latency_ms,
            "iteration_count": iteration_count,
        },
    }
