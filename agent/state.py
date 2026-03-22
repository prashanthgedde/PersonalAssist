import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str
    chat_id: int
    tool_calls: list | int
    sources: list
    final_response: str
    metadata: dict
    should_use_tools: bool
    iteration_count: int
