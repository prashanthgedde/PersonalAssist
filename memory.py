import logging
import os
from datetime import datetime, timezone

from mem0 import Memory

BASE_SYSTEM_PROMPT = (
    "You are a helpful personal assistant with web access. "
    "You are provided with the current date, time, and user location at the start of every message — use them when answering questions about time, scheduling, or location. "
    "Always use the search_web tool when the user asks about news, current events, prices, scores, or anything time-sensitive.\n\n"
    "Format all responses using Telegram HTML. Supported tags:\n"
    "- <b>bold</b> for headings or key terms\n"
    "- <i>italic</i> for emphasis\n"
    "- <code>code</code> for values, tickers, commands\n"
    "- <a href='url'>text</a> for links\n"
    "For news or lists, use one item per line starting with a bullet (•). Keep responses concise and scannable on mobile."
)

_mem = None


def _get_mem() -> Memory:
    global _mem
    if _mem is None:
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "personalassist",
                    "path": os.getenv("CHROMA_PATH", "./chroma_db")
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            }
        }
        _mem = Memory.from_config(config)
        logging.info("mem0 initialized with local Chroma store")
    return _mem


def build_system_prompt(chat_id: int, query: str = "") -> str:
    """Build system prompt injecting current datetime and memories relevant to the current query."""
    tz_name = os.getenv("USER_TIMEZONE", "UTC")
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    datetime_line = f"Current date and time: {now.strftime('%A, %B %d, %Y %H:%M %Z')}"

    location = os.getenv("USER_LOCATION", "")
    location_line = f"User's location: {location}" if location else ""

    context = ""
    if query:
        try:
            results = _get_mem().search(query, user_id=str(chat_id), limit=5)
            memories = [r["memory"] for r in results.get("results", [])]
            if memories:
                context = "\n".join(f"- {m}" for m in memories)
        except Exception as e:
            logging.error(f"Memory search failed: {e}")

    prompt = f"{BASE_SYSTEM_PROMPT}\n\n{datetime_line}"
    if location_line:
        prompt += f"\n{location_line}"
    if context:
        prompt += f"\n\nRelevant things you know about this user:\n{context}"
    return prompt


def add_to_memory(chat_id: int, messages: list):
    """Extract and persist memories from a user/assistant exchange."""
    try:
        _get_mem().add(messages, user_id=str(chat_id))
        logging.info(f"Memory updated for {chat_id}")
    except Exception as e:
        logging.error(f"Memory add failed: {e}")
