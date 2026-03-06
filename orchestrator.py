import json
import logging

from openai import OpenAI

_ORCHESTRATOR_SYSTEM = (
    "Classify whether the user's request requires simple or complex processing.\n\n"
    "Simple: single lookup, weather, stock price, set a reminder, casual chat, one-step question.\n"
    "Complex: multi-step research, comparative analysis, tasks needing multiple sequential searches, "
    "comprehensive reports, 'find everything about X', 'compare A vs B', 'summarize and analyze'.\n\n"
    "Reply with exactly one word: simple or complex"
)

MAX_AGENTIC_ITERATIONS = 6


def classify_query(client: OpenAI, user_text: str) -> str:
    """Returns 'simple' or 'complex'."""
    try:
        resp = client.chat.completions.create(
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
            logging.info(f"Orchestrator: COMPLEX — '{user_text[:80]}'")
            return "complex"
    except Exception as e:
        logging.warning(f"Orchestrator classification failed, defaulting to simple: {e}")

    logging.info(f"Orchestrator: SIMPLE — '{user_text[:80]}'")
    return "simple"


def run_agentic_loop(
    client: OpenAI,
    messages: list,
    tool_definitions: list,
    tool_fns: dict,
) -> str:
    """
    Multi-step agentic loop: the LLM can call tools repeatedly across multiple
    rounds until it produces a final answer or MAX_AGENTIC_ITERATIONS is reached.

    tool_fns: dict mapping tool name -> callable (kwargs-based dispatch)
    Mutates `messages` in-place so the caller's history stays current.
    Returns the final assistant text.
    """
    for iteration in range(MAX_AGENTIC_ITERATIONS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # LLM finished — no more tool calls
            messages.append({"role": "assistant", "content": msg.content})
            logging.info(f"Agentic loop done after {iteration + 1} iteration(s)")
            return msg.content

        # Execute all tool calls in this round
        messages.append(msg.model_dump())
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = tool_fns.get(fn_name)
            if fn:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = "Unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        logging.info(f"Agentic loop iteration {iteration + 1} complete")

    # Max iterations hit — ask LLM to wrap up with what it has
    logging.warning("Agentic loop hit max iterations, requesting final summary")
    messages.append({
        "role": "user",
        "content": "Please summarize everything you've found and give a final answer now.",
    })
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    final = response.choices[0].message.content
    messages.append({"role": "assistant", "content": final})
    return final
