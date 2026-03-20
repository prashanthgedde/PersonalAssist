import asyncio
import os
import logging

import logging_config

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


async def test_agent():
    from agent.simple_agent import run_agent

    test_queries = [
        "What's the weather in London?",
        "What's Apple's stock price?",
        "Hello, how are you?",
        "Search for latest AI news",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = run_agent(chat_id=12345, user_query=query)

        print(f"\nResult dict: {result}")
        print(f"\nFinal Response: {result.get('final_response', 'N/A')}")
        print(f"\nSources: {result.get('sources', [])}")
        print(f"\nMetadata: {result.get('metadata', {})}")
        print()


if __name__ == "__main__":
    asyncio.run(test_agent())
