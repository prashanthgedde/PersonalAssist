import asyncio
import logging
import os
import re
from datetime import datetime

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

from agent.graph import run_agent_streaming
from response_formatter import format_response

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8080))
STREAM_CHUNK_SIZE = 50


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    logging.info(f"[HANDLE] Processing message from chat_id={chat_id}")
    logging.info(f"[HANDLE] User query: {user_text[:100]}...")

    accumulated_response = ""
    final_sources = []
    final_metadata = {}
    message_id = None

    try:
        async for event in run_agent_streaming(chat_id=chat_id, user_query=user_text):
            event_type = event.get("type")
            content = event.get("content", "")

            if event_type == "tools_start":
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")

            elif event_type == "stream":
                accumulated_response = content
                if len(content) >= STREAM_CHUNK_SIZE:
                    try:
                        if message_id is None:
                            sent = await update.message.reply_text(content[:4096])
                            message_id = sent.message_id
                        else:
                            sent = await context.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=message_id,
                                text=content[:4096],
                            )
                            message_id = sent.message_id
                    except Exception as e:
                        logging.debug(f"[STREAM] Could not update message: {e}")

            elif event_type == "respond":
                accumulated_response = content
                final_sources = event.get("sources", [])
                final_metadata = event.get("metadata", {})

            elif event_type == "done":
                accumulated_response = content

        bot_text = accumulated_response or "No response generated"

        logging.info(f"[RESPONSE] Raw LLM output (len={len(bot_text)})")
        logging.debug(f"[RESPONSE] Raw text:\n{bot_text}\n--- END RAW ---")

        formatted_text, parse_mode = format_response(bot_text, final_sources, final_metadata)

        logging.info(
            f"[RESPONSE] Formatted text (len={len(formatted_text)}), parse_mode={parse_mode}"
        )

        try:
            if message_id is not None:
                await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass

        if parse_mode:
            await update.message.reply_text(formatted_text, parse_mode=parse_mode)
        else:
            await update.message.reply_text(formatted_text)
        logging.info(f"[RESPONSE] Successfully sent to Telegram")

    except Exception as e:
        logging.error(f"[HANDLE] Streaming failed: {e}")
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await asyncio.sleep(0.5)
        try:
            await update.message.reply_text(
                f"Sorry, I encountered an error processing your request."
            )
        except Exception:
            pass


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
