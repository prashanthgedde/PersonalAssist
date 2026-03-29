from typing import Protocol, runtime_checkable


class Message:
    """A single conversation turn."""

    def __init__(self, role: str, content: str):
        self.role = role  # "human" or "assistant"
        self.content = content

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content[:60]!r})"


@runtime_checkable
class MemoryBackend(Protocol):
    """
    Abstract contract for all memory backends.

    Implementations:
    - in_memory.py   — ephemeral dict (dev/testing)
    - markdown.py    — markdown files on disk (current default)
    - sqlite.py      — SQLite (future)
    - vector.py      — vector DB (future)
    """

    def get_history(self, chat_id: int, limit: int) -> list[Message]:
        """Return the last `limit` conversation turns for this chat."""
        ...

    def save_turn(self, chat_id: int, human: str, assistant: str) -> None:
        """Persist a single human→assistant exchange."""
        ...

    def get_long_term(self, chat_id: int) -> str:
        """Return the long-term context blob for this user (may be empty string)."""
        ...

    def save_long_term(self, chat_id: int, content: str) -> None:
        """Overwrite the long-term context blob for this user."""
        ...
