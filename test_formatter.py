from response_formatter import format_as_html


def test_html_formatter():
    test_cases = [
        {
            "name": "Simple text response",
            "text": "Hello! How can I help you today?",
            "sources": [],
            "metadata": {},
        },
        {
            "name": "Response with sources",
            "text": "The latest AI news shows significant developments in LLMs.",
            "sources": [
                {
                    "title": "OpenAI Announces GPT-5",
                    "url": "https://openai.com/blog/gpt5",
                },
                {
                    "title": "Google DeepMind Research",
                    "url": "https://deepmind.com/research",
                },
            ],
            "metadata": {"tool_calls": 1, "total_latency_ms": 1500},
        },
        {
            "name": "Response with markdown",
            "text": "**Stock Price**: $150.00\n*Up 2.5%* from yesterday.\nUse `AAPL` to reference this ticker.",
            "sources": [],
            "metadata": {"tool_calls": 1, "total_latency_ms": 800},
        },
        {
            "name": "Full response with all metadata",
            "text": "Here's a summary of the latest tech news.",
            "sources": [
                {"title": "TechCrunch", "url": "https://techcrunch.com"},
                {"title": "The Verge", "url": "https://theverge.com"},
            ],
            "metadata": {
                "tool_calls": 2,
                "total_latency_ms": 3500,
                "iteration_count": 3,
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Test: {tc['name']}")
        print("=" * 60)

        html = format_as_html(tc["text"], tc.get("sources", []), tc.get("metadata", {}))

        print(f"HTML Output:\n{html}")
        print()


if __name__ == "__main__":
    test_html_formatter()
