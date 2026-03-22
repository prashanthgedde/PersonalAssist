from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import (
    build_system_prompt,
    first_respond_node,
    respond_node,
    should_continue_tools,
)
from agent.state import AgentState


class TestBuildSystemPrompt:
    def test_build_system_prompt_contains_datetime(self):
        with patch.dict("os.environ", {"USER_TIMEZONE": "UTC"}):
            prompt = build_system_prompt()
            assert "Current date and time" in prompt
            assert "You are a helpful personal assistant" in prompt

    def test_build_system_prompt_with_location(self):
        with patch.dict("os.environ", {"USER_TIMEZONE": "UTC", "USER_LOCATION": "New York"}):
            prompt = build_system_prompt()
            assert "User's location: New York" in prompt

    def test_build_system_prompt_markdown_formatting(self):
        prompt = build_system_prompt()
        assert "Telegram MarkdownV2" in prompt
        assert "*bold*" in prompt
        assert "_italic_" in prompt


def create_mock_tool_call(name: str, args: dict, tool_id: str = "call_123"):
    return {
        "name": name,
        "args": args,
        "id": tool_id,
        "type": "tool_call",
    }


class TestShouldContinueTools:
    def test_continue_when_tool_calls_present(self):
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="test",
                    tool_calls=[create_mock_tool_call("get_weather", {"location": "London"})],
                )
            ],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }
        result = should_continue_tools(state)
        assert result == "continue"

    def test_respond_when_no_tool_calls(self):
        state: AgentState = {
            "messages": [AIMessage(content="test")],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }
        result = should_continue_tools(state)
        assert result == "respond"

    def test_respond_when_max_iterations_reached(self):
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="test",
                    tool_calls=[create_mock_tool_call("get_weather", {"location": "London"})],
                )
            ],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 10,
        }
        result = should_continue_tools(state)
        assert result == "respond"


class TestFirstRespondNode:
    def test_sets_should_use_tools_when_tool_calls_present(self):
        mock_response = AIMessage(
            content="",
            tool_calls=[create_mock_tool_call("get_weather", {"location": "London"})],
        )
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        state: AgentState = {
            "messages": [],
            "user_query": "What's the weather?",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }

        with patch("agent.nodes.get_llm", return_value=mock_llm):
            result = first_respond_node(state)

        assert result["should_use_tools"] is True
        assert result["iteration_count"] == 1

    def test_sets_final_response_when_no_tools_needed(self):
        mock_response = AIMessage(content="Hello, how can I help?")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        state: AgentState = {
            "messages": [],
            "user_query": "Hello!",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }

        with patch("agent.nodes.get_llm", return_value=mock_llm):
            result = first_respond_node(state)

        assert result["should_use_tools"] is False
        assert "Hello, how can I help?" in result["final_response"]


class TestRespondNode:
    def test_generates_final_response(self):
        mock_response = AIMessage(content="Final answer here.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        state: AgentState = {
            "messages": [HumanMessage(content="test")],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": [],
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }

        with patch("agent.nodes.get_llm", return_value=mock_llm):
            result = respond_node(state)

        assert "Final answer here." in result["final_response"]
        assert "total_latency_ms" in result["metadata"]
