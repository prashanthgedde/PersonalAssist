import asyncio
import json
import logging
import os
import re
import time

# MUST import logging_config first to set up DEBUG logging before other modules
import logging_config  # noqa: F401

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from tools import search_web, get_stock, get_weather, TOOL_DEFINITIONS
from memory import build_system_prompt, add_to_memory
from reminders import init_scheduler, set_reminder
from orchestrator import classify_query, run_agentic_loop
from response_formatter import format_response
from response_summary import ResponseSummary

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://personalassist.fly.dev
PORT = int(os.getenv("PORT", 8080))

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
user_history = {}


async def post_init(application: Application) -> None:
    """Called after the event loop starts — safe place to init the scheduler."""
    init_scheduler(application.bot)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Create summary tracker for this response
    summary = ResponseSummary(chat_id=chat_id, query=user_text)

    # Show typing indicator immediately before any blocking calls
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Refresh system prompt — runs embedding search in thread so it doesn't block the event loop
    logging.info(f"[SYSTEM_PROMPT] Building system prompt with mem0 search...")
    loop = asyncio.get_event_loop()
    prompt_start = time.time()
    system_content = await loop.run_in_executor(None, build_system_prompt, chat_id, user_text)
    prompt_time_ms = (time.time() - prompt_start) * 1000
    logging.info(f"[SYSTEM_PROMPT] Ready in {prompt_time_ms:.1f}ms")
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
    summary.classification = complexity
    logging.info(f"[ORCHESTRATOR] Query complexity: {complexity}")

    if complexity == "complex":
        # Agentic loop: LLM can call tools across multiple rounds
        logging.info(f"[ORCHESTRATOR] Using agentic loop")
        bot_text = await run_agentic_loop(client, user_history[chat_id], TOOL_DEFINITIONS, tool_fns, summary=summary)
        logging.info(f"[ORCHESTRATOR] Agentic loop returned text (len={len(bot_text)})")
    else:
        # Fast path: single OpenAI call + at most one round of tool calls
        logging.info(f"[ORCHESTRATOR] Using fast path")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_history[chat_id],
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        logging.info(f"[ORCHESTRATOR] First response: has_tool_calls={bool(response_message.tool_calls)}")

        if response_message.tool_calls:
            logging.info(f"[ORCHESTRATOR] Tool calls: {[tc.function.name for tc in response_message.tool_calls]}")
            user_history[chat_id].append(response_message.model_dump())

            for tool_call in response_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn = tool_fns.get(fn_name)
                logging.debug(f"[ORCHESTRATOR] Tool {fn_name} called with args: {fn_args}")
                tool_start = time.time()
                content = fn(**fn_args) if fn else "Unknown tool."
                tool_latency_ms = (time.time() - tool_start) * 1000
                logging.info(f"[ORCHESTRATOR] Tool {fn_name} returned: {str(content)[:100]}... ({tool_latency_ms:.1f}ms)")
                logging.debug(f"[ORCHESTRATOR] {fn_name} full result:\n{content}")
                # Record in summary
                summary.add_tool_call(fn_name, fn_args, content, sources=fn_args.get("sources"), latency_ms=tool_latency_ms)
                user_history[chat_id].append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

            logging.info(f"[ORCHESTRATOR] Making second request with tool results")
            second_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=user_history[chat_id]
            )
            bot_text = second_response.choices[0].message.content
            logging.info(f"[ORCHESTRATOR] Second response text (len={len(bot_text)})")
        else:
            bot_text = response_message.content
            logging.info(f"[ORCHESTRATOR] Direct response text (len={len(bot_text)})")
        user_history[chat_id].append({"role": "assistant", "content": bot_text})

    # Log raw LLM response before formatting
    summary.lm_response_len = len(bot_text)
    logging.info(f"[RESPONSE] Raw LLM output (len={len(bot_text)})")
    logging.debug(f"[RESPONSE] Raw text:\n{bot_text}\n--- END RAW ---")

    # Format and send reply immediately — don't wait for memory persistence
    formatted_text, parse_mode = await format_response(bot_text)
    summary.formatting = parse_mode or "plain"
    summary.final_response_len = len(formatted_text)

    logging.info(f"[RESPONSE] Formatted text (len={len(formatted_text)}), parse_mode={parse_mode}")
    logging.debug(f"[RESPONSE] Final formatted:\n{formatted_text}\n--- END FORMATTED ---")

    try:
        if parse_mode:
            await update.message.reply_text(formatted_text, parse_mode=parse_mode)
        else:
            await update.message.reply_text(formatted_text)
        logging.info(f"[RESPONSE] Successfully sent to Telegram")
        summary.status = "sent"
    except Exception as e:
        logging.error(f"[RESPONSE] Failed to send message: {e}")
        logging.debug(f"[RESPONSE] Problematic text: {formatted_text[:500]}")
        summary.error = str(e)
        # Last resort: send plain text with NO formatting
        try:
            stripped = re.sub(r'[*_`\[\]()]', '', formatted_text)
            logging.info(f"[RESPONSE] Retrying with stripped text (len={len(stripped)})")
            await update.message.reply_text(stripped)
            logging.info(f"[RESPONSE] Successfully sent stripped version")
            summary.status = "sent_fallback"
        except Exception as e2:
            logging.error(f"[RESPONSE] Failed even with stripped text: {e2}")
            summary.status = "failed"
            summary.error = str(e2)

    # Persist to mem0 in the background so it doesn't block the response
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, add_to_memory, chat_id, [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot_text}
    ])

    # Log consolidated summary
    summary.log()


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
