import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from tools import search_web, get_stock, get_weather, TOOL_DEFINITIONS
from memory import build_system_prompt, add_to_memory
from reminders import init_scheduler, set_reminder
from orchestrator import classify_query, run_agentic_loop

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://personalassist.fly.dev
PORT = int(os.getenv("PORT", 8080))

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
user_history = {}


async def post_init(application: Application) -> None:
    """Called after the event loop starts — safe place to init the scheduler."""
    init_scheduler(application.bot)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Show typing indicator immediately before any blocking calls
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Refresh system prompt — runs embedding search in thread so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    system_content = await loop.run_in_executor(None, build_system_prompt, chat_id, user_text)
    if chat_id not in user_history:
        user_history[chat_id] = [{"role": "system", "content": system_content}]
    else:
        user_history[chat_id][0] = {"role": "system", "content": system_content}

    user_history[chat_id].append({"role": "user", "content": user_text})

    # Tool dispatch table — set_reminder needs chat_id bound at call time
    tool_fns = {
        "search_web": search_web,
        "get_stock": get_stock,
        "get_weather": get_weather,
        "set_reminder": lambda **kw: set_reminder(chat_id, kw["message"], kw["remind_at"]),
    }

    # Orchestrator: decide fast path vs agentic loop
    complexity = await classify_query(client, user_text)

    if complexity == "complex":
        # Agentic loop: LLM can call tools across multiple rounds
        bot_text = await run_agentic_loop(client, user_history[chat_id], TOOL_DEFINITIONS, tool_fns)
    else:
        # Fast path: single OpenAI call + at most one round of tool calls
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_history[chat_id],
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )
        response_message = response.choices[0].message

        if response_message.tool_calls:
            user_history[chat_id].append(response_message.model_dump())

            for tool_call in response_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn = tool_fns.get(fn_name)
                content = fn(**fn_args) if fn else "Unknown tool."
                user_history[chat_id].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

            second_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=user_history[chat_id]
            )
            bot_text = second_response.choices[0].message.content
        else:
            bot_text = response_message.content
        user_history[chat_id].append({"role": "assistant", "content": bot_text})

    # Send reply immediately — don't wait for memory persistence
    try:
        await update.message.reply_text(bot_text, parse_mode=ParseMode.HTML)
    except Exception:
        await update.message.reply_text(bot_text)

    # Persist to mem0 in the background so it doesn't block the response
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, add_to_memory, chat_id, [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot_text}
    ])


if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    if WEBHOOK_URL:
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}",
        )
    else:
        app.run_polling()                       # fallback for local dev
