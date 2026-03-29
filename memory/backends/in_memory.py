from collections import defaultdict

from memory.protocol import Message


class InMemoryBackend:
    """
    Ephemeral in-process backend. Data is lost on restart.
    Use for development, testing, or as a drop-in for the old behaviour.
    """

    def __init__(self) -> None:
        self._history: dict[int, list[Message]] = defaultdict(list)
        self._long_term: dict[int, str] = {}

    def get_history(self, chat_id: int, limit: int) -> list[Message]:
        turns = self._history[chat_id]
        return turns[-limit:] if limit else turns[:]

    def save_turn(self, chat_id: int, human: str, assistant: str) -> None:
        self._history[chat_id].append(Message(role="human", content=human))
        self._history[chat_id].append(Message(role="assistant", content=assistant))

    def get_long_term(self, chat_id: int) -> str:
        return self._long_term.get(chat_id, "")

    def save_long_term(self, chat_id: int, content: str) -> None:
        self._long_term[chat_id] = content
