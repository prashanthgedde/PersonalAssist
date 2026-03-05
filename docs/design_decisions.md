# PersonalAssist - Design Decisions

## 1. Modular File Structure

**Decision**: Separate code into `main.py`, `tools.py`, and `memory.py`.

**Rationale**: Keep orchestration logic (main.py) clean and focused on the bot loop. Tool functions and their OpenAI schema definitions live together in tools.py so they stay in sync. Memory logic is isolated in memory.py so the backend can be swapped without touching the rest.

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

## 5. Tool: search_web (DuckDuckGo)

**Decision**: Use `DDGS.news()` as primary, fall back to `DDGS.text()` if no news results.

**Rationale**: `ddgs.news()` returns dated news articles with titles and snippets — ideal for current events. `ddgs.text()` is a broader web search useful for factual queries that aren't news-based. No API key required (free).

---

## 6. Tool: get_weather (wttr.in)

**Decision**: Use wttr.in free JSON API (`?format=j1`) with `timeout=(5, 20)`.

**Rationale**: No API key required for personal use. The tuple timeout `(connect_timeout, read_timeout)` is necessary because wttr.in occasionally has slow reads; a flat `timeout=10` caused read timeout errors.

**Alternative if reliability is a concern**: OpenWeatherMap (free tier, requires API key).

---

## 7. Tool: get_stock (yfinance)

**Decision**: Use yfinance to fetch `t.info` dict for price and key metrics.

**Rationale**: No API key required for personal use. Returns `currentPrice` (or `regularMarketPrice` as fallback), change%, market cap, 52-week high/low.

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
