from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
import operator


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    chat_id: int
    tool_calls: list
    sources: list
    final_response: str
    metadata: dict
    should_use_tools: bool
    iteration_count: int
