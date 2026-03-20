import asyncio
import logging
import os
import re

import logging_config  # noqa: F401

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.simple_agent import run_agent
from response_formatter import format_response

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8080))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    logging.info(f"[HANDLE] Processing message from chat_id={chat_id}")
    logging.info(f"[HANDLE] User query: {user_text[:100]}...")

    result = run_agent(chat_id=chat_id, user_query=user_text)

    bot_text = result.get("final_response", "No response generated")
    sources = result.get("sources", [])
    metadata = result.get("metadata", {})

    logging.info(f"[RESPONSE] Raw LLM output (len={len(bot_text)})")
    logging.debug(f"[RESPONSE] Raw text:\n{bot_text}\n--- END RAW ---")

    formatted_text, parse_mode = format_response(bot_text, sources, metadata)

    logging.info(
        f"[RESPONSE] Formatted text (len={len(formatted_text)}), parse_mode={parse_mode}"
    )
    logging.debug(
        f"[RESPONSE] Final formatted:\n{formatted_text}\n--- END FORMATTED ---"
    )

    try:
        if parse_mode:
            await update.message.reply_text(formatted_text, parse_mode=parse_mode)
        else:
            await update.message.reply_text(formatted_text)
        logging.info(f"[RESPONSE] Successfully sent to Telegram")
    except Exception as e:
        logging.error(f"[RESPONSE] Failed to send message: {e}")
        logging.debug(f"[RESPONSE] Problematic text: {formatted_text[:500]}")
        try:
            stripped = re.sub(r"[*_`\[\]()]", "", formatted_text)
            stripped = re.sub(r"<[^>]+>", "", stripped)
            logging.info(
                f"[RESPONSE] Retrying with stripped text (len={len(stripped)})"
            )
            await update.message.reply_text(stripped)
            logging.info(f"[RESPONSE] Successfully sent stripped version")
        except Exception as e2:
            logging.error(f"[RESPONSE] Failed even with stripped text: {e2}")


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
