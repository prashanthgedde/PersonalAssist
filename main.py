import asyncio
import logging
import os
from contextlib import suppress

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

import logging_config  # noqa: F401
from agent.multi_turn_agent import run_agent
from memory.backends.markdown import MarkdownBackend
from memory.manager import MemoryManager
from response_formatter import format_response

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8080))

memory = MemoryManager(
    backend=MarkdownBackend(root="data/memory"),
    history_window=20,
    long_term_enabled=True,
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    logging.info(f"[HANDLE] Processing message from chat_id={chat_id}")
    logging.info(f"[HANDLE] User query: {user_text[:100]}...")

    try:
        logging.info("[MAIN] Calling run_agent")
        result = run_agent(chat_id=chat_id, user_query=user_text, memory=memory)
        logging.info(f"[MAIN] Got result type: {type(result)}")

        bot_text = result.get("final_response", "No response generated")
        final_sources = result.get("sources", [])
        final_metadata = result.get("metadata", {})

        logging.info(f"[RESPONSE] Raw LLM output (len={len(bot_text)})")
        logging.debug(f"[RESPONSE] Raw text:\n{bot_text}\n--- END RAW ---")

        formatted_text, parse_mode = format_response(bot_text, final_sources, final_metadata)

        logging.info(
            f"[RESPONSE] Formatted text (len={len(formatted_text)}), parse_mode={parse_mode}"
        )

        if parse_mode:
            await update.message.reply_text(formatted_text, parse_mode=parse_mode)
        else:
            await update.message.reply_text(formatted_text)
        logging.info("[RESPONSE] Successfully sent to Telegram")

    except Exception as e:
        logging.error(f"[HANDLE] Agent failed: {e}")
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await asyncio.sleep(0.5)
        with suppress(Exception):
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request."
            )


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    if WEBHOOK_URL:
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}",
        )
    else:
        app.run_polling()
