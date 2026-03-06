import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

_scheduler = None
_bot = None


def init_scheduler(bot):
    global _scheduler, _bot
    _bot = bot
    db_path = os.path.join(os.getenv("CHROMA_PATH", "./chroma_db"), "reminders.db")
    jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
    _scheduler = AsyncIOScheduler(jobstores=jobstores)
    _scheduler.start()
    logging.info("Reminder scheduler started")


async def _send_reminder(chat_id: int, message: str):
    await _bot.send_message(chat_id=chat_id, text=f"Reminder: {message}")


def set_reminder(chat_id: int, message: str, remind_at: str) -> str:
    try:
        dt = datetime.fromisoformat(remind_at)
        _scheduler.add_job(
            _send_reminder,
            trigger="date",
            run_date=dt,
            args=[chat_id, message]
        )
        return f"Reminder set for {dt.strftime('%A, %b %d at %I:%M %p')}"
    except Exception as e:
        logging.error(f"Failed to set reminder: {e}")
        return f"Failed to set reminder: {e}"
