import json
import logging
import os

MEMORY_FILE = "user_memory.json"
SUMMARIZE_THRESHOLD = 20  # messages before summarizing
KEEP_RECENT = 6           # messages to retain after summarization

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant with web access. "
    "Always use the search_web tool when the user asks about news, current events, "
    "prices, scores, or anything time-sensitive. "
    "Use save_memory to remember important facts or preferences the user shares."
)


def _load() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {}


def _save(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_fact(chat_id: int, fact: str):
    data = _load()
    key = str(chat_id)
    if key not in data:
        data[key] = {"facts": []}
    if fact not in data[key]["facts"]:
        data[key]["facts"].append(fact)
        _save(data)
        logging.info(f"Saved memory for {chat_id}: {fact}")


def build_system_prompt(chat_id: int) -> str:
    facts = _load().get(str(chat_id), {}).get("facts", [])
    if not facts:
        return BASE_SYSTEM_PROMPT
    facts_str = "\n".join(f"- {f}" for f in facts)
    return f"{BASE_SYSTEM_PROMPT}\n\nWhat you know about this user:\n{facts_str}"


def maybe_summarize(chat_id: int, history: list, client) -> list:
    """Summarize older messages when history grows too long."""
    non_system = [m for m in history if m.get("role") != "system"]
    if len(non_system) < SUMMARIZE_THRESHOLD:
        return history

    to_summarize = non_system[:-KEEP_RECENT]
    recent = non_system[-KEEP_RECENT:]

    logging.info(f"Summarizing {len(to_summarize)} messages for {chat_id}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely, preserving key facts, decisions, and context."},
                {"role": "user", "content": json.dumps(to_summarize)}
            ]
        )
        summary = response.choices[0].message.content
        system_msg = next((m for m in history if m.get("role") == "system"), None)
        new_history = []
        if system_msg:
            new_history.append(system_msg)
        new_history.append({"role": "assistant", "content": f"[Conversation summary: {summary}]"})
        new_history.extend(recent)
        return new_history
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return history
