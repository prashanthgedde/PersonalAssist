import logging
from datetime import datetime, timezone
from pathlib import Path

from memory.protocol import Message

logger = logging.getLogger(__name__)

# Max number of turns kept in the history file before oldest are trimmed.
_MAX_STORED_TURNS = 100


class MarkdownBackend:
    """
    Disk-backed memory using plain markdown files.

    Directory layout (one sub-folder per chat_id):
        <root>/
        └── <chat_id>/
            ├── history.md      — chronological conversation log
            └── long_term.md    — free-text user profile / facts

    history.md format
    -----------------
    Each turn is a fenced block so it is human-readable and easy to parse:

        <!-- turn:1 ts:2024-01-01T12:00:00Z -->
        **Human:** hello there
        **Assistant:** hi! how can I help?

    long_term.md format
    -------------------
    Free-form markdown written/rewritten by the LLM or directly by the user.
    No internal structure is enforced — the whole file is read and injected as-is.
    """

    _TURN_SEP = "<!-- turn:"

    def __init__(self, root: str | Path = "data/memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_dir(self, chat_id: int) -> Path:
        d = self.root / str(chat_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _history_path(self, chat_id: int) -> Path:
        return self._chat_dir(chat_id) / "history.md"

    def _long_term_path(self, chat_id: int) -> Path:
        return self._chat_dir(chat_id) / "long_term.md"

    def _parse_history(self, path: Path) -> list[Message]:
        """Parse history.md into a flat list of Message objects."""
        if not path.exists():
            return []

        messages: list[Message] = []
        raw = path.read_text(encoding="utf-8")
        # Split on turn separator comments
        blocks = raw.split(self._TURN_SEP)
        for block in blocks[1:]:  # skip preamble before first turn
            lines = block.strip().splitlines()
            human_lines: list[str] = []
            assistant_lines: list[str] = []
            current: list[str] | None = None
            for line in lines:
                if line.startswith("**Human:**"):
                    current = human_lines
                    rest = line[len("**Human:**"):].strip()
                    if rest:
                        current.append(rest)
                elif line.startswith("**Assistant:**"):
                    current = assistant_lines
                    rest = line[len("**Assistant:**"):].strip()
                    if rest:
                        current.append(rest)
                elif current is not None:
                    current.append(line)
            if human_lines:
                messages.append(Message(role="human", content="\n".join(human_lines).strip()))
            if assistant_lines:
                messages.append(
                    Message(role="assistant", content="\n".join(assistant_lines).strip())
                )
        return messages

    def _write_history(self, path: Path, messages: list[Message]) -> None:
        """Serialise message list back to history.md."""
        lines: list[str] = []
        turn_index = 1
        # Messages come in human/assistant pairs
        i = 0
        while i < len(messages):
            human_msg = messages[i] if i < len(messages) else None
            assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"{self._TURN_SEP}{turn_index} ts:{ts} -->")
            if human_msg:
                lines.append(f"**Human:** {human_msg.content}")
            if assistant_msg:
                lines.append(f"**Assistant:** {assistant_msg.content}")
            lines.append("")
            turn_index += 1
            i += 2
        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # MemoryBackend protocol implementation
    # ------------------------------------------------------------------

    def get_history(self, chat_id: int, limit: int) -> list[Message]:
        path = self._history_path(chat_id)
        messages = self._parse_history(path)
        # limit counts individual messages (not turns), return last `limit`
        return messages[-limit:] if limit else messages[:]

    def save_turn(self, chat_id: int, human: str, assistant: str) -> None:
        path = self._history_path(chat_id)
        messages = self._parse_history(path)
        messages.append(Message(role="human", content=human))
        messages.append(Message(role="assistant", content=assistant))

        # Trim oldest turns if over the stored cap (pairs of messages)
        max_messages = _MAX_STORED_TURNS * 2
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
            logger.debug(f"[MEMORY] Trimmed history for chat_id={chat_id} to {_MAX_STORED_TURNS} turns")

        self._write_history(path, messages)
        logger.debug(f"[MEMORY] Saved turn for chat_id={chat_id}, total messages={len(messages)}")

    def get_long_term(self, chat_id: int) -> str:
        path = self._long_term_path(chat_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def save_long_term(self, chat_id: int, content: str) -> None:
        path = self._long_term_path(chat_id)
        path.write_text(content.strip(), encoding="utf-8")
        logger.debug(f"[MEMORY] Saved long-term context for chat_id={chat_id}")
