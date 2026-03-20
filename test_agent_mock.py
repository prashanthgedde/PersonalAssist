import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.simple_agent import run_agent


def create_mock_response(content: str = "", tool_calls: list = None):
    mock = AIMessage(content=content)
    if tool_calls:
        mock.tool_calls = tool_calls
    return mock


class MockTool:
    def invoke(self, args):
        return "Mock tool result"


class TestSimpleAgent:
    def test_greeting_no_tools(self):
        """Test simple greeting without any tool calls"""
        mock_response = create_mock_response(content="Hello! How can I help you today?")

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        with patch("agent.simple_agent.get_llm", return_value=mock_llm):
            result = run_agent(chat_id=12345, user_query="Hello!")

        assert result["final_response"] == "Hello! How can I help you today?"
        assert result["metadata"]["tool_calls"] == 0
        assert result["sources"] == []

    def test_tool_call_weather(self):
        """Test weather query triggers get_weather tool"""
        mock_response_with_tool = create_mock_response(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "London"},
                    "id": "call_123",
                }
            ],
        )
        mock_final_response = create_mock_response(
            content="The weather in London is sunny, 18°C."
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.side_effect = [
            mock_response_with_tool,
            mock_final_response,
        ]

        mock_llm.bind_tools.return_value.invoke.return_value = mock_response_with_tool

        with (
            patch("agent.simple_agent.get_llm", return_value=mock_llm),
            patch("agent.simple_agent.TOOL_MAP", {"get_weather": MockTool()}),
        ):
            result = run_agent(
                chat_id=12345, user_query="What's the weather in London?"
            )

        # First call returns tool call, second call returns final response
        assert result["metadata"]["tool_calls"] == 1

    def test_tool_call_stock(self):
        """Test stock query triggers get_stock tool"""
        mock_response_with_tool = create_mock_response(
            content="",
            tool_calls=[
                {"name": "get_stock", "args": {"symbol": "AAPL"}, "id": "call_456"}
            ],
        )
        mock_final_response = create_mock_response(
            content="Apple (AAPL) is trading at $175.00"
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.side_effect = [
            mock_response_with_tool,
            mock_final_response,
        ]

        with (
            patch("agent.simple_agent.get_llm", return_value=mock_llm),
            patch("agent.simple_agent.TOOL_MAP", {"get_stock": MockTool()}),
        ):
            result = run_agent(chat_id=12345, user_query="What's Apple's stock price?")

        assert result["metadata"]["tool_calls"] == 1

    def test_multiple_tool_calls(self):
        """Test query requiring multiple tool calls"""
        mock_response_with_tools = create_mock_response(
            content="",
            tool_calls=[
                {"name": "search_web", "args": {"query": "AI news"}, "id": "call_1"},
                {"name": "get_weather", "args": {"location": "NYC"}, "id": "call_2"},
            ],
        )
        mock_final_response = create_mock_response(
            content="Here's your summary with news and weather."
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.side_effect = [
            mock_response_with_tools,
            mock_final_response,
        ]

        with (
            patch("agent.simple_agent.get_llm", return_value=mock_llm),
            patch(
                "agent.simple_agent.TOOL_MAP",
                {"search_web": MockTool(), "get_weather": MockTool()},
            ),
        ):
            result = run_agent(chat_id=12345, user_query="AI news and NYC weather")

        assert result["metadata"]["tool_calls"] == 2

    def test_search_web_extracts_sources(self):
        """Test that search_web tool extracts sources from results"""
        mock_response_with_tool = create_mock_response(
            content="",
            tool_calls=[
                {"name": "search_web", "args": {"query": "test"}, "id": "call_789"}
            ],
        )
        mock_final_response = create_mock_response(content="Here are the results.")

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.side_effect = [
            mock_response_with_tool,
            mock_final_response,
        ]

        class MockSearchTool:
            def invoke(self, args):
                return "Check out [OpenAI](https://openai.com) and [Google](https://google.com)"

        with (
            patch("agent.simple_agent.get_llm", return_value=mock_llm),
            patch("agent.simple_agent.TOOL_MAP", {"search_web": MockSearchTool()}),
        ):
            result = run_agent(chat_id=12345, user_query="Search for something")

        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "OpenAI"

    def test_metadata_includes_latency(self):
        """Test that metadata includes latency"""
        mock_response = create_mock_response(content="Response text")

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        with patch("agent.simple_agent.get_llm", return_value=mock_llm):
            result = run_agent(chat_id=12345, user_query="Hello")

        assert "total_latency_ms" in result["metadata"]
        assert result["metadata"]["total_latency_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
