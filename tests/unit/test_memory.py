import tempfile
from pathlib import Path

from memory.backends.in_memory import InMemoryBackend
from memory.backends.markdown import MarkdownBackend
from memory.manager import MemoryManager

# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class TestInMemoryBackend:
    def setup_method(self):
        self.backend = InMemoryBackend()

    def test_get_history_empty(self):
        assert self.backend.get_history(1, limit=10) == []

    def test_save_and_get_history(self):
        self.backend.save_turn(1, "hello", "hi there")
        msgs = self.backend.get_history(1, limit=10)
        assert len(msgs) == 2
        assert msgs[0].role == "human"
        assert msgs[0].content == "hello"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "hi there"

    def test_history_limit(self):
        for i in range(5):
            self.backend.save_turn(1, f"q{i}", f"a{i}")
        msgs = self.backend.get_history(1, limit=4)
        assert len(msgs) == 4
        # Should return the most recent 4
        assert msgs[0].content == "q3"

    def test_history_isolated_per_chat(self):
        self.backend.save_turn(1, "chat1", "response1")
        self.backend.save_turn(2, "chat2", "response2")
        assert len(self.backend.get_history(1, limit=10)) == 2
        assert len(self.backend.get_history(2, limit=10)) == 2
        assert self.backend.get_history(1, limit=10)[0].content == "chat1"

    def test_long_term_empty_by_default(self):
        assert self.backend.get_long_term(1) == ""

    def test_save_and_get_long_term(self):
        self.backend.save_long_term(1, "User prefers metric units.")
        assert self.backend.get_long_term(1) == "User prefers metric units."

    def test_long_term_overwrite(self):
        self.backend.save_long_term(1, "old content")
        self.backend.save_long_term(1, "new content")
        assert self.backend.get_long_term(1) == "new content"

    def test_long_term_isolated_per_chat(self):
        self.backend.save_long_term(1, "chat1 facts")
        self.backend.save_long_term(2, "chat2 facts")
        assert self.backend.get_long_term(1) == "chat1 facts"
        assert self.backend.get_long_term(2) == "chat2 facts"


# ---------------------------------------------------------------------------
# MarkdownBackend
# ---------------------------------------------------------------------------


class TestMarkdownBackend:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.backend = MarkdownBackend(root=self._tmpdir.name)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_get_history_no_file(self):
        assert self.backend.get_history(42, limit=10) == []

    def test_save_and_get_history(self):
        self.backend.save_turn(1, "what is the weather?", "It is sunny.")
        msgs = self.backend.get_history(1, limit=10)
        assert len(msgs) == 2
        assert msgs[0].role == "human"
        assert msgs[0].content == "what is the weather?"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "It is sunny."

    def test_history_persists_across_instances(self):
        self.backend.save_turn(1, "hello", "world")
        # New instance pointing at the same directory
        backend2 = MarkdownBackend(root=self._tmpdir.name)
        msgs = backend2.get_history(1, limit=10)
        assert len(msgs) == 2
        assert msgs[0].content == "hello"

    def test_multiple_turns_order(self):
        self.backend.save_turn(1, "first", "reply1")
        self.backend.save_turn(1, "second", "reply2")
        msgs = self.backend.get_history(1, limit=10)
        assert len(msgs) == 4
        assert msgs[0].content == "first"
        assert msgs[2].content == "second"

    def test_history_limit(self):
        for i in range(5):
            self.backend.save_turn(1, f"q{i}", f"a{i}")
        msgs = self.backend.get_history(1, limit=4)
        assert len(msgs) == 4
        assert msgs[0].content == "q3"

    def test_history_file_is_human_readable(self):
        self.backend.save_turn(1, "hello", "hi there")
        path = Path(self._tmpdir.name) / "1" / "history.md"
        contents = path.read_text()
        assert "**Human:**" in contents
        assert "**Assistant:**" in contents
        assert "hello" in contents

    def test_long_term_empty_when_no_file(self):
        assert self.backend.get_long_term(99) == ""

    def test_save_and_get_long_term(self):
        self.backend.save_long_term(1, "Prefers Celsius.")
        assert self.backend.get_long_term(1) == "Prefers Celsius."

    def test_long_term_persists_across_instances(self):
        self.backend.save_long_term(1, "Lives in Tokyo.")
        backend2 = MarkdownBackend(root=self._tmpdir.name)
        assert backend2.get_long_term(1) == "Lives in Tokyo."

    def test_long_term_overwrite(self):
        self.backend.save_long_term(1, "old")
        self.backend.save_long_term(1, "new")
        assert self.backend.get_long_term(1) == "new"

    def test_history_trim_at_cap(self):
        from memory.backends.markdown import _MAX_STORED_TURNS
        # Save more turns than the cap
        for i in range(_MAX_STORED_TURNS + 5):
            self.backend.save_turn(1, f"q{i}", f"a{i}")
        msgs = self.backend.get_history(1, limit=0)  # 0 = no limit
        assert len(msgs) <= _MAX_STORED_TURNS * 2


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def setup_method(self):
        self.backend = InMemoryBackend()
        self.manager = MemoryManager(
            backend=self.backend,
            history_window=10,
            long_term_enabled=True,
        )

    def test_load_context_empty(self):
        ctx = self.manager.load_context(1, "hello")
        assert ctx.history == []
        assert ctx.long_term == ""

    def test_save_and_load_history(self):
        self.manager.save_context(1, "ping", "pong")
        ctx = self.manager.load_context(1, "next question")
        assert len(ctx.history) == 2

    def test_history_is_langchain_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage
        self.manager.save_context(1, "hello", "hi")
        ctx = self.manager.load_context(1, "follow up")
        assert isinstance(ctx.history[0], HumanMessage)
        assert isinstance(ctx.history[1], AIMessage)

    def test_long_term_injected_when_present(self):
        self.backend.save_long_term(1, "User is called Alice.")
        ctx = self.manager.load_context(1, "who am I?")
        assert ctx.long_term == "User is called Alice."

    def test_long_term_disabled(self):
        manager = MemoryManager(
            backend=self.backend,
            history_window=10,
            long_term_enabled=False,
        )
        self.backend.save_long_term(1, "some facts")
        ctx = manager.load_context(1, "query")
        assert ctx.long_term == ""

    def test_save_long_term_via_manager(self):
        self.manager.save_long_term(1, "Likes dark mode.")
        assert self.backend.get_long_term(1) == "Likes dark mode."

    def test_save_long_term_ignored_when_disabled(self):
        manager = MemoryManager(
            backend=self.backend,
            history_window=10,
            long_term_enabled=False,
        )
        manager.save_long_term(1, "should not be saved")
        assert self.backend.get_long_term(1) == ""

    def test_history_window_respected(self):
        manager = MemoryManager(backend=self.backend, history_window=4)
        for i in range(5):
            manager.save_context(1, f"q{i}", f"a{i}")
        ctx = manager.load_context(1, "new")
        assert len(ctx.history) == 4
