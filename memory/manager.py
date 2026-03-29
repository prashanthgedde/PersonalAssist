import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory.protocol import MemoryBackend, Message

logger = logging.getLogger(__name__)

# Default sliding window: number of individual messages (not turns) to inject.
DEFAULT_HISTORY_WINDOW = 20  # 10 turns


@dataclass
class LoadedContext:
    """What the manager returns to run_agent() before each call."""

    history: list  # list of LangChain HumanMessage / AIMessage
    long_term: str  # raw text blob, empty string if none


class MemoryManager:
    """
    Orchestrates short-term (sliding window) and long-term (text blob) memory.

    Usage
    -----
    manager = MemoryManager(backend=MarkdownBackend())

    # Before the agent runs:
    ctx = manager.load_context(chat_id, user_query)

    # After the agent returns:
    manager.save_context(chat_id, user_query, final_response)
    """

    def __init__(
        self,
        backend: MemoryBackend,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        long_term_enabled: bool = True,
    ) -> None:
        self.backend = backend
        self.history_window = history_window
        self.long_term_enabled = long_term_enabled

    # ------------------------------------------------------------------
    # Public API used by run_agent()
    # ------------------------------------------------------------------

    def load_context(self, chat_id: int, query: str) -> LoadedContext:
        """
        Load history + long-term context for a given chat before the agent runs.

        Returns a LoadedContext with:
        - history: LangChain message objects ready to be inserted into messages[]
        - long_term: raw string to inject into the system prompt (empty if disabled)
        """
        raw_messages = self.backend.get_history(chat_id, self.history_window)
        lc_messages = _to_langchain_messages(raw_messages)

        long_term = ""
        if self.long_term_enabled:
            long_term = self.backend.get_long_term(chat_id)

        logger.debug(
            f"[MEMORY] Loaded context for chat_id={chat_id}: "
            f"{len(lc_messages)} history messages, "
            f"long_term={'yes' if long_term else 'no'}"
        )
        return LoadedContext(history=lc_messages, long_term=long_term)

    def save_context(self, chat_id: int, human: str, assistant: str) -> None:
        """Persist the latest exchange after the agent returns."""
        self.backend.save_turn(chat_id, human, assistant)
        logger.debug(f"[MEMORY] Saved turn for chat_id={chat_id}")

    def save_long_term(self, chat_id: int, content: str) -> None:
        """Overwrite the long-term context blob (called externally or after summarisation)."""
        if self.long_term_enabled:
            self.backend.save_long_term(chat_id, content)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_langchain_messages(messages: list[Message]) -> list:
    result = []
    for msg in messages:
        if msg.role == "human":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
    return result
