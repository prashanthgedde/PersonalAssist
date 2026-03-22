from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import create_agent_graph, run_agent, tools_node_wrapper
from agent.state import AgentState


def create_mock_tool_call(name: str, args: dict = None, tool_id: str = "call_123"):
    return {
        "name": name,
        "args": args or {},
        "id": tool_id,
        "type": "tool_call",
    }


class TestAgentGraph:
    def test_create_agent_graph(self):
        graph = create_agent_graph()
        assert graph is not None

    def test_tools_node_wrapper_processes_messages(self):
        mock_response = AIMessage(
            content="Weather result",
            name="get_weather",
            tool_call_id="call_123",
        )

        mock_tool_node = MagicMock()
        mock_tool_node.invoke.return_value = [mock_response]

        state: AgentState = {
            "messages": [
                HumanMessage(content="What's the weather?"),
                AIMessage(
                    content="",
                    tool_calls=[create_mock_tool_call("get_weather", {"location": "London"})],
                ),
            ],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": 0,
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }

        with patch("agent.graph.get_tool_node", return_value=mock_tool_node):
            result = tools_node_wrapper(state)

        assert "messages" in result
        assert "iteration_count" in result
        assert result["iteration_count"] == 1

    def test_tools_node_wrapper_extracts_sources_from_search(self):
        mock_response = AIMessage(
            content="Check out [OpenAI](https://openai.com) and [Google](https://google.com)",
            name="search_web",
            tool_call_id="call_456",
        )

        mock_tool_node = MagicMock()
        mock_tool_node.invoke.return_value = [mock_response]

        state: AgentState = {
            "messages": [
                HumanMessage(content="Search for AI news"),
                AIMessage(
                    content="",
                    tool_calls=[create_mock_tool_call("search_web", {"query": "AI news"})],
                ),
            ],
            "user_query": "",
            "chat_id": 1,
            "tool_calls": 0,
            "sources": [],
            "final_response": "",
            "metadata": {},
            "should_use_tools": False,
            "iteration_count": 0,
        }

        with patch("agent.graph.get_tool_node", return_value=mock_tool_node):
            result = tools_node_wrapper(state)

        assert len(result["sources"]) == 2


class TestRunAgent:
    def test_run_agent_returns_response(self):
        mock_response = AIMessage(content="Hello, I'm your assistant!")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        with patch("agent.nodes.get_llm", return_value=mock_llm):
            result = run_agent(chat_id=12345, user_query="Hello")

        assert "final_response" in result

    def test_run_agent_with_initial_state(self):
        mock_response = AIMessage(content="Response")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        with patch("agent.nodes.get_llm", return_value=mock_llm):
            result = run_agent(
                chat_id=12345,
                user_query="Test query",
                config={"thread_id": "thread_1"},
            )

        assert "final_response" in result
