import json
import logging
import time

from openai import AsyncOpenAI

# Import logging_config to ensure DEBUG level is available
import logging_config  # noqa: F401

_ORCHESTRATOR_SYSTEM = (
    "Classify whether the user's request requires simple or complex processing.\n\n"
    "Simple: single lookup, weather, stock price, set a reminder, casual chat, one-step question.\n"
    "Complex: multi-step research, comparative analysis, tasks needing multiple sequential searches, "
    "comprehensive reports, 'find everything about X', 'compare A vs B', 'summarize and analyze'.\n\n"
    "Reply with exactly one word: simple or complex"
)

MAX_AGENTIC_ITERATIONS = 6

# Keywords that guarantee a simple fast-path without an LLM classify call
_SIMPLE_PATTERNS = (
    "weather", "stock", "price", "remind", "reminder", "what time", "what day",
    "what's the", "whats the", "hello", "hi ", "hey ", "thanks", "thank you",
    "who are you", "what can you", "help", "how are you",
)

# Keywords that guarantee complex routing without an LLM classify call
_COMPLEX_PATTERNS = (
    "research", "compare", "comparison", "comprehensive", "summarize and",
    "analyze", "find everything", "all sources", "deep dive", "in depth",
    "pros and cons", "versus", " vs ", "multiple sources",
)


async def classify_query(client: AsyncOpenAI, user_text: str) -> str:
    """Returns 'simple' or 'complex'. Uses keyword heuristics first to avoid an LLM round trip."""
    lower = user_text.lower()

    for kw in _SIMPLE_PATTERNS:
        if kw in lower:
            logging.info(f"Orchestrator: SIMPLE (heuristic) — '{user_text[:80]}'")
            return "simple"

    for kw in _COMPLEX_PATTERNS:
        if kw in lower:
            logging.info(f"Orchestrator: COMPLEX (heuristic) — '{user_text[:80]}'")
            return "complex"

    # Ambiguous — ask the LLM
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            max_tokens=5,
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        if "complex" in verdict:
            logging.info(f"Orchestrator: COMPLEX (LLM) — '{user_text[:80]}'")
            return "complex"
    except Exception as e:
        logging.warning(f"Orchestrator classification failed, defaulting to simple: {e}")

    logging.info(f"Orchestrator: SIMPLE (LLM) — '{user_text[:80]}'")
    return "simple"


async def run_agentic_loop(
    client: AsyncOpenAI,
    messages: list,
    tool_definitions: list,
    tool_fns: dict,
    summary=None,
) -> str:
    """
    Multi-step agentic loop: the LLM can call tools repeatedly across multiple
    rounds until it produces a final answer or MAX_AGENTIC_ITERATIONS is reached.

    tool_fns: dict mapping tool name -> callable (kwargs-based dispatch)
    summary: optional ResponseSummary to track tool calls and iterations
    Mutates `messages` in-place so the caller's history stays current.
    Returns the final assistant text.
    """
    for iteration in range(MAX_AGENTIC_ITERATIONS):
        logging.info(f"[AGENTIC] Iteration {iteration + 1}/{MAX_AGENTIC_ITERATIONS}")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # LLM finished — no more tool calls
            messages.append({"role": "assistant", "content": msg.content})
            if summary:
                summary.agentic_loops = iteration + 1
            logging.info(f"[AGENTIC] Loop done after {iteration + 1} iteration(s), response len={len(msg.content)}")
            logging.debug(f"[AGENTIC] Final response:\n{msg.content}\n--- END ---")
            return msg.content

        # Execute all tool calls in this round
        tool_names = [tc.function.name for tc in msg.tool_calls]
        logging.info(f"[AGENTIC] Iteration {iteration + 1}: tool calls: {tool_names}")
        messages.append(msg.model_dump())

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = tool_fns.get(fn_name)
            logging.debug(f"[AGENTIC] Tool {fn_name} called with args: {fn_args}")
            tool_start = time.time()
            if fn:
                try:
                    result = fn(**fn_args)
                    tool_latency_ms = (time.time() - tool_start) * 1000
                    logging.info(f"[AGENTIC] Tool {fn_name} succeeded: {str(result)[:100]}... ({tool_latency_ms:.1f}ms)")
                    logging.debug(f"[AGENTIC] {fn_name} full result:\n{result}")
                except Exception as e:
                    tool_latency_ms = (time.time() - tool_start) * 1000
                    result = f"Tool error: {e}"
                    logging.error(f"[AGENTIC] Tool {fn_name} failed: {e} ({tool_latency_ms:.1f}ms)")
            else:
                tool_latency_ms = (time.time() - tool_start) * 1000
                result = "Unknown tool."
                logging.warning(f"[AGENTIC] Unknown tool: {fn_name}")

            # Record in summary
            if summary:
                summary.add_tool_call(fn_name, fn_args, result, sources=fn_args.get("sources"), latency_ms=tool_latency_ms)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        logging.info(f"[AGENTIC] Iteration {iteration + 1} complete")

    # Max iterations hit — ask LLM to wrap up with what it has
    logging.warning("[AGENTIC] Hit max iterations, requesting final summary")
    messages.append({
        "role": "user",
        "content": "Please summarize everything you've found and give a final answer now.",
    })
    response = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    final = response.choices[0].message.content
    messages.append({"role": "assistant", "content": final})
    logging.info(f"[AGENTIC] Max iterations summary response len={len(final)}")
    logging.debug(f"[AGENTIC] Summary response:\n{final}\n--- END ---")
    return final
