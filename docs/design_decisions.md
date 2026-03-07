# PersonalAssist - Design Decisions

## 1. Modular File Structure

**Decision**: Separate code into `main.py`, `tools.py`, `memory.py`, `reminders.py`, and `orchestrator.py`.

**Rationale**: Keep orchestration logic (main.py) clean and focused on routing. Tool functions and their OpenAI schema definitions live together in tools.py so they stay in sync. Memory logic is isolated in memory.py so the backend can be swapped without touching the rest. Reminders and orchestration are their own modules to keep concerns separated.

---

## 2. OpenAI Tool Calling Pattern

**Decision**: Use `tool_choice="auto"` with a descriptive system prompt that explicitly instructs the LLM to use tools for time-sensitive queries.

**Rationale**: `tool_choice="required"` forces a tool call on every message, which is wasteful for conversational exchanges. Instead, the system prompt instructs the model to always use `search_web` for news, prices, scores, or anything time-sensitive. This gives the LLM flexibility for general conversation while ensuring tools are used when needed.

**Key implementation detail**: `response_message.model_dump()` must be used when appending the assistant message containing tool calls to `user_history`. The OpenAI Python SDK returns Pydantic objects, not plain dicts — appending the object directly causes `AttributeError` on `.get()` calls.

---

## 3. Memory Backend: mem0 OSS (Local)

**Decision**: Use mem0 OSS with a local Chroma vector store over (a) hand-rolled JSON memory, (b) mem0 cloud.

**Rationale**:
- Hand-rolled memory required manual `save_memory` tool calls and keyword-based summarization — fragile and required explicit LLM instruction to save facts.
- mem0 auto-extracts and deduplicates facts from every conversation via its own LLM pass — no tool call needed.
- Local mode (Chroma at `./chroma_db/`) avoids cloud API keys, costs, and data privacy concerns for a personal assistant.
- mem0 cloud was not chosen to keep everything self-hosted for personal use.

**Config**: `gpt-4o-mini` for LLM, `text-embedding-3-small` for embeddings, Chroma file store.

---

## 4. Per-Message Memory Injection

**Decision**: Rebuild the system prompt on every message using semantic search against stored memories.

**Rationale**: Rather than loading all memories (which grows unbounded), search for the top-5 most relevant memories to the current query. This keeps the context window lean and ensures only contextually relevant facts are injected. The system prompt's first element (`user_history[chat_id][0]`) is replaced each turn.

---

## 5. Tool: search_web (Tavily primary, DuckDuckGo fallback)

**Decision**: Use Tavily as the primary search provider, with DuckDuckGo as a fallback if Tavily is unavailable or fails.

**Rationale**: Tavily is an AI-optimized search API that returns cleaner, more structured results than scraping DuckDuckGo. It supports `include_domains` for restricting searches to specific sources (e.g., `reddit.com`, `x.com`, `youtube.com`). DuckDuckGo (`DDGS.news()` → `DDGS.text()`) is retained as a zero-config fallback.

**Config**: `TAVILY_API_KEY` env var required for Tavily. If unset, silently falls back to DuckDuckGo.

---

## 6. Tool: get_weather (wttr.in)

**Decision**: Use wttr.in free JSON API (`?format=j1`) with `timeout=(5, 20)`.

**Rationale**: No API key required for personal use. The tuple timeout `(connect_timeout, read_timeout)` is necessary because wttr.in occasionally has slow reads; a flat `timeout=10` caused read timeout errors.

**Alternative if reliability is a concern**: OpenWeatherMap (free tier, requires API key).

---

## 7. Tool: get_stock (yfinance)

**Decision**: Use yfinance to fetch `t.info` dict for price and key metrics.

**Rationale**: No API key required for personal use. Returns `currentPrice` (or `regularMarketPrice` as fallback), change%, market cap, 52-week high/low. Same library will be reused for portfolio tracking.

---

## 8. Dependency Constraint: onnxruntime < 1.21

**Decision**: Pin `onnxruntime<1.21` in pyproject.toml.

**Rationale**: chromadb (pulled in by mem0ai) depends on onnxruntime. Versions ≥1.21 dropped Python 3.10 wheel support. Pinning to <1.21 ensures compatibility with Python 3.10 (the project's runtime).

---

## 9. In-memory Conversation History

**Decision**: Store per-user conversation history in a Python dict (`user_history`) keyed by `chat_id`.

**Rationale**: Simple, fast, sufficient for a personal single-user bot. History is lost on restart, which is acceptable because long-term facts are persisted via mem0. No database needed.

---

## 10. No save_memory Tool

**Decision**: Removed the explicit `save_memory` tool after switching to mem0.

**Rationale**: mem0 automatically extracts and stores memories from every `add()` call using its internal LLM pass. Requiring the LLM to explicitly call a save tool was brittle — it could forget, or save irrelevant things. Auto-extraction is more reliable.

---

## 11. Reminders: APScheduler + SQLite on Fly Volume

**Decision**: Use `AsyncIOScheduler` from APScheduler with `SQLAlchemyJobStore` backed by SQLite, stored on the Fly.io persistent volume at `$CHROMA_PATH/reminders.db`.

**Rationale**: APScheduler integrates cleanly with asyncio (required by python-telegram-bot). SQLite persistence means reminders survive bot restarts. Storing the DB inside `CHROMA_PATH` keeps all persistent data co-located on the same fly.io volume — no second volume needed.

**Key implementation detail**: `init_scheduler()` must be called inside `post_init` (python-telegram-bot's async hook), not at module import time. Calling it at import time causes `RuntimeError: no running event loop` because APScheduler's `AsyncIOScheduler` needs a running event loop to start.

---

## 12. Orchestrator: Fast Path vs Agentic Loop

**Decision**: Add a lightweight classifier (`classify_query`) that routes each message to either a single-pass flow or a multi-step agentic loop.

**Rationale**: Most queries (weather, stock, reminders, casual chat) need only one round of tool calls. Routing all messages through a multi-step loop adds latency and cost for no benefit. For complex queries (research, comparison, multi-source summarization), the agentic loop allows the LLM to call tools across multiple iterations — each round informed by the last — up to `MAX_AGENTIC_ITERATIONS = 6`.

**Implementation**: `orchestrator.py` — `classify_query()` makes a cheap `max_tokens=5, temperature=0` call to gpt-4o-mini. `run_agentic_loop()` implements a ReAct-style loop that mutates `user_history` in place. Falls back to simple on classifier error.

**Tool dispatch**: Replaced the if/elif chain with a `tool_fns` dict (kwargs-based). `set_reminder` is wrapped in a lambda to bind `chat_id` at call time, keeping the dict pattern uniform.

---

## 13. Telegram HTML Formatting

**Decision**: Format all LLM responses using Telegram HTML (`ParseMode.HTML`) with a plain-text fallback.

**Rationale**: Telegram supports a subset of HTML for rich formatting (`<b>`, `<i>`, `<code>`, `<a href>`). Instructing the LLM to use these tags produces mobile-friendly, scannable output. A try/except fallback (`reply_text(bot_text)` without parse_mode) prevents crashes if the LLM produces malformed HTML.

---

## 14. Second Brain / Notes (Planned)

**Decision (planned)**: Store user notes as markdown files on the fly.io volume, indexed into a separate Chroma collection for semantic search.

**Rationale**:
- Markdown files are human-readable, portable, and easy to back up via `flyctl ssh`.
- A separate Chroma collection (`notes`) keeps explicit user-written thoughts distinct from auto-extracted conversation memories (mem0 collection).
- Notes will be injected into the system prompt alongside mem0 memories — surfaced semantically per query — so they passively influence every response without the user needing to reference them explicitly.
- Planned organization: `notes/daily/YYYY-MM-DD.md` for brain dumps, `notes/topics/<topic>.md` for longer-form notes.
- Tools: `add_note`, `search_notes`, `get_daily_note`.

---

## 15. Portfolio Tracking (Planned)

**Decision (planned)**: Store holdings as a JSON file on the fly.io volume; fetch prices via yfinance (already installed).

**Rationale**: No broker OAuth or API keys needed — yfinance uses public Yahoo Finance data (~15 min delayed). Holdings (ticker, shares, avg cost) stored locally in JSON for privacy. Tools: `get_portfolio` (P&L summary), `update_portfolio` (add/remove position).
