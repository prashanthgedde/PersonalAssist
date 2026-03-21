import os
import logging

import logging_config

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def test_multi_turn_agent():
    """Test the MULTI-TURN agentic loop.

    Flow: LLM -> tools -> LLM -> tools -> ... -> LLM -> done

    Key advantage: Can call MORE tools after seeing results from previous tools.
    """
    from agent.multi_turn_agent import run_agent, MAX_ITERATIONS

    test_queries = [
        ("What's the weather in London?", "Single tool", 2),
        ("What's Apple's stock price?", "Single tool", 2),
        ("Hello, how are you?", "No tools", 1),
        ("Search for latest AI news", "Single search", 2),
        (
            "What's the weather in Tokyo and search for what's happening there?",
            "Parallel tools",
            2,
        ),
        ("Compare the stock price of Tesla and Apple", "Parallel tools", 2),
        (
            "What is the current CEO of Apple? Find the latest news about that CEO.",
            "Parallel tools",
            2,
        ),
        (
            "Get the weather for San Francisco, then tell me about what's happening there.",
            "Sequential",
            2,
        ),
        ("Compare the stock prices of NVDA, AMD, and Intel.", "Triple parallel", 2),
        ("What's the weather in Paris, London, and Tokyo?", "Triple parallel", 2),
        (
            "Find the latest news about the stock market, then tell me how it affects Apple shares.",
            "DEPENDENT (multi-turn)",
            3,
        ),
    ]

    print("\n" + "=" * 70)
    print("MULTI-TURN AGENT (Iterative)")
    print("=" * 70)
    print(
        f"Flow: LLM -> tools -> LLM -> tools -> ... (max {MAX_ITERATIONS} iterations)"
    )
    print("Advantage: Can call MORE tools after seeing results\n")

    for query, description, expected_iters in test_queries:
        print(f"\n--- {description} ---")
        print(f"Query: {query}")
        print("-" * 50)

        result = run_agent(chat_id=12345, user_query=query)

        actual_iters = result["metadata"]["iteration_count"]
        status = (
            "✓" if actual_iters == expected_iters else f"✗ (expected {expected_iters})"
        )

        print(f"Iterations: {actual_iters} {status}")
        print(f"Tools called: {result['metadata']['tool_calls']}")
        print(f"Response: {result['final_response'][:150]}...")


def compare_agents():
    """Compare simple vs multi-turn on a query that requires iteration."""
    from agent.simple_agent import run_agent as simple_run
    from agent.multi_turn_agent import run_agent as multi_run

    query = "Find the latest news about the stock market, then tell me how it affects Apple shares."

    print("\n" + "=" * 70)
    print("COMPARISON: Simple vs Multi-Turn")
    print("=" * 70)
    print(f"Query: {query}")
    print("\nThis query REQUIRES multi-turn because:")
    print("  1. First need stock market news")
    print("  2. Then use that info to determine how it affects Apple")
    print("-" * 70)

    print("\n[SIMPLE AGENT - Single Round]")
    result = simple_run(chat_id=12345, user_query=query)
    print(f"Iterations: {result['metadata']['iteration_count']}")
    print(f"Tools called: {result['metadata']['tool_calls']}")
    print(f"Response: {result['final_response'][:300]}...")

    print("\n[MULTI-TURN AGENT - Iterative]")
    result = multi_run(chat_id=12345, user_query=query)
    print(f"Iterations: {result['metadata']['iteration_count']}")
    print(f"Tools called: {result['metadata']['tool_calls']}")
    print(f"Response: {result['final_response'][:300]}...")


if __name__ == "__main__":
    test_multi_turn_agent()
    compare_agents()
