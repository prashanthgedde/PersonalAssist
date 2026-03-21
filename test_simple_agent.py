import os
import logging

import logging_config

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def test_simple_agent():
    """Test the SIMPLE single-round agent.

    Flow: LLM -> tools (one shot) -> LLM -> done

    Limitation: Cannot call more tools after seeing results.
    """
    from agent.simple_agent import run_agent

    test_queries = [
        ("What's the weather in London?", "Single tool"),
        ("What's Apple's stock price?", "Single tool"),
        ("Hello, how are you?", "No tools"),
        ("Search for latest AI news", "Single search"),
        (
            "What's the weather in Tokyo and search for what's happening there?",
            "Parallel tools",
        ),
        ("Compare the stock price of Tesla and Apple", "Parallel tools"),
    ]

    print("\n" + "=" * 70)
    print("SIMPLE AGENT (Single Round)")
    print("=" * 70)
    print("Flow: LLM -> tools (all at once) -> LLM -> done")
    print("Limitation: Cannot call MORE tools after seeing results\n")

    for query, description in test_queries:
        print(f"\n--- {description} ---")
        print(f"Query: {query}")
        print("-" * 50)

        result = run_agent(chat_id=12345, user_query=query)

        print(f"Iterations: {result['metadata']['iteration_count']}")
        print(f"Tools called: {result['metadata']['tool_calls']}")
        print(f"Response: {result['final_response'][:150]}...")


if __name__ == "__main__":
    test_simple_agent()
