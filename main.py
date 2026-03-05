import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from tools import search_web, get_stock, get_weather, TOOL_DEFINITIONS
from memory import build_system_prompt, add_to_memory

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
user_history = {}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Refresh system prompt with memories relevant to this message
    system_content = build_system_prompt(chat_id, user_text)
    if chat_id not in user_history:
        user_history[chat_id] = [{"role": "system", "content": system_content}]
    else:
        user_history[chat_id][0] = {"role": "system", "content": system_content}

    user_history[chat_id].append({"role": "user", "content": user_text})
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Step 1: Send to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=user_history[chat_id],
        tools=TOOL_DEFINITIONS,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Step 2: Handle tool calls
    if response_message.tool_calls:
        user_history[chat_id].append(response_message.model_dump())

        for tool_call in response_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "search_web":
                content = search_web(fn_args["query"])
            elif fn_name == "get_stock":
                content = get_stock(fn_args["ticker"])
            elif fn_name == "get_weather":
                content = get_weather(fn_args["location"])
            else:
                content = "Unknown tool."

            user_history[chat_id].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content
            })

        # Step 3: Final response with tool results
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_history[chat_id]
        )
        bot_text = second_response.choices[0].message.content
    else:
        bot_text = response_message.content

    user_history[chat_id].append({"role": "assistant", "content": bot_text})

    # Persist this exchange to mem0
    add_to_memory(chat_id, [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": bot_text}
    ])

    await update.message.reply_text(bot_text)


if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()
