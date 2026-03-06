import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import reminders


@pytest.fixture(autouse=True)
async def reset_reminders():
    """Reset module state between tests."""
    reminders._scheduler = None
    reminders._bot = None
    yield
    if reminders._scheduler and reminders._scheduler.running:
        try:
            reminders._scheduler.shutdown(wait=False)
        except Exception:
            pass
    reminders._scheduler = None
    reminders._bot = None


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.send_message = AsyncMock()
    return bot


@pytest.mark.asyncio
async def test_init_scheduler(mock_bot, tmp_path):
    with patch.dict("os.environ", {"CHROMA_PATH": str(tmp_path)}):
        reminders.init_scheduler(mock_bot)

    assert reminders._scheduler is not None
    assert reminders._scheduler.running
    assert reminders._bot is mock_bot


@pytest.mark.asyncio
async def test_set_reminder_returns_confirmation(mock_bot, tmp_path):
    with patch.dict("os.environ", {"CHROMA_PATH": str(tmp_path)}):
        reminders.init_scheduler(mock_bot)

    future = datetime.now() + timedelta(hours=1)
    result = reminders.set_reminder(12345, "Buy groceries", future.isoformat())

    assert "Reminder set" in result
    assert reminders._scheduler.get_jobs()  # job was scheduled


@pytest.mark.asyncio
async def test_set_reminder_invalid_datetime(mock_bot, tmp_path):
    with patch.dict("os.environ", {"CHROMA_PATH": str(tmp_path)}):
        reminders.init_scheduler(mock_bot)

    result = reminders.set_reminder(12345, "Test", "not-a-date")

    assert "Failed to set reminder" in result
    assert not reminders._scheduler.get_jobs()  # no job added


@pytest.mark.asyncio
async def test_send_reminder_calls_bot(mock_bot, tmp_path):
    with patch.dict("os.environ", {"CHROMA_PATH": str(tmp_path)}):
        reminders.init_scheduler(mock_bot)

    await reminders._send_reminder(12345, "Buy groceries")

    mock_bot.send_message.assert_called_once_with(
        chat_id=12345,
        text="Reminder: Buy groceries"
    )
