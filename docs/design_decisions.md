# PersonalAssist - Design Decisions

## 1. Modular Agent Structure

**Decision**: Separate agents into `simple_agent.py` (single-round) and `multi_turn_agent.py` (iterative).

**Rationale**: Having both implementations available helps with learning and comparison. The simple agent demonstrates the basic tool-calling pattern, while the multi-turn agent shows how iterative tool calling works. They can be swapped in `main.py` depending on use case.

**File structure**:
```
agent/
├── simple_agent.py         # Single-round: LLM → tools → LLM → done
├── multi_turn_agent.py     # Iterative: LLM → tools → LLM → tools → ... → done
├── tools.py                # Tool definitions (search, stock, weather)
└── graph.py                # LangGraph implementation (separate)
```

---

## 2. Simple Agent (Single-Round)

**Decision**: `simple_agent.py` implements a single-round tool calling pattern.

**Flow**:
```
┌─────────────────────────────────────────────────────┐
│ 1. LLM call (with tools bound)                      │
│         ↓                                           │
│ 2. Execute ALL tools in parallel (one shot)         │
│         ↓                                           │
│ 3. LLM call with results → final response           │
└─────────────────────────────────────────────────────┘
```

**Limitation**: Cannot call MORE tools after seeing results. If the query requires chaining (e.g., "search X, then use X to find Y"), this fails.

---

## 3. Multi-Turn Agent (Iterative)

**Decision**: `multi_turn_agent.py` implements an iterative agentic loop.

**Flow**:
```
┌─────────────────────────────────────────────────────┐
│ while iteration < MAX_ITERATIONS (6):               │
│                                                     │
│   1. LLM call with tools                            │
│          ↓                                          │
│   2. No tool_calls? → break, return response        │
│          ↓                                          │
│   3. Execute tools, accumulate messages             │
│          ↓                                          │
│   4. Loop back to step 1                           │
└─────────────────────────────────────────────────────┘
```

**Advantage**: Can chain dependent tool calls. Example:
- Query: "Find stock market news, then tell me how it affects Apple"
- Iteration 1: `search_web` (market news)
- Iteration 2: `get_stock` (AAPL) - uses context from iteration 1
- Iteration 3: Final response

**Trade-off**: More LLM calls = higher latency and cost.

---

## 4. Choosing Between Simple and Multi-Turn

**Decision**: Use `simple_agent` for most queries, `multi_turn_agent` for complex chaining.

| Use Case | Recommended Agent |
|----------|------------------|
| Weather, stock, single search | `simple_agent` |
| Parallel independent tools | `simple_agent` |
| Dependent tool calls (A→B) | `multi_turn_agent` |
| Research/analysis tasks | `multi_turn_agent` |

---

## 5. OpenAI Tool Calling Pattern

**Decision**: Use `tool_choice="auto"` with a descriptive system prompt that explicitly instructs the LLM to use tools for time-sensitive queries.

**Rationale**: `tool_choice="required"` forces a tool call on every message, which is wasteful for conversational exchanges. Instead, the system prompt instructs the model to always use `search_web` for news, prices, scores, or anything time-sensitive.

**Key implementation detail**: `response.tool_calls` returns a list of dicts with `name`, `args`, and `id`. Always use `.get()` to access these fields.

---

## 6. Tool: search_web (Tavily primary, DuckDuckGo fallback)

**Decision**: Use Tavily as the primary search provider, with DuckDuckGo as a fallback.

**Rationale**: Tavily is an AI-optimized search API that returns cleaner, more structured results. It supports `include_domains` for restricting searches to specific sources (e.g., `reddit.com`, `x.com`). DuckDuckGo is retained as a zero-config fallback.

**Config**: `TAVILY_API_KEY` env var. Falls back silently if unset.

---

## 7. Tool: get_weather (wttr.in)

**Decision**: Use wttr.in free JSON API (`?format=j1`) with `timeout=(5, 20)`.

**Rationale**: No API key required. The tuple timeout `(connect_timeout, read_timeout)` is necessary because wttr.in occasionally has slow reads.

**Alternative**: OpenWeatherMap (free tier, requires API key).

---

## 8. Tool: get_stock (yfinance)

**Decision**: Use yfinance to fetch `t.info` dict for price and key metrics.

**Rationale**: No API key required. Returns `currentPrice`, change%, market cap, 52-week high/low. Same library will be reused for portfolio tracking.

---

## 9. Telegram HTML Formatting

**Decision**: Format all LLM responses using Telegram HTML (`ParseMode.HTML`) with a plain-text fallback.

**Rationale**: Telegram supports a subset of HTML (`<b>`, `<i>`, `<code>`, `<a href>`). Instructing the LLM to use these tags produces mobile-friendly, scannable output. A try/except fallback prevents crashes if the LLM produces malformed HTML.

---

## 10. Per-Message Memory Injection

**Decision**: Rebuild the system prompt on every message using semantic search against stored memories.

**Rationale**: Rather than loading all memories (which grows unbounded), search for the top-5 most relevant memories to the current query. This keeps the context window lean.

---

## 11. In-memory Conversation History

**Decision**: Store per-user conversation history in a Python dict (`user_history`) keyed by `chat_id`.

**Rationale**: Simple, fast, sufficient for a personal single-user bot. History is lost on restart, which is acceptable because long-term facts are persisted via mem0.

---

## 12. Reminders: APScheduler + SQLite on Fly Volume

**Decision**: Use `AsyncIOScheduler` from APScheduler with `SQLAlchemyJobStore` backed by SQLite, stored on the Fly.io persistent volume.

**Rationale**: APScheduler integrates cleanly with asyncio (required by python-telegram-bot). SQLite persistence means reminders survive bot restarts.

**Key implementation detail**: `init_scheduler()` must be called inside `post_init` (python-telegram-bot's async hook), not at module import time.

---

## 13. Dependency Constraint: onnxruntime < 1.21

**Decision**: Pin `onnxruntime<1.21` in pyproject.toml.

**Rationale**: chromadb (pulled in by mem0ai) depends on onnxruntime. Versions ≥1.21 dropped Python 3.10 wheel support.

---

## 14. Second Brain / Notes (Planned)

**Decision (planned)**: Store user notes as markdown files on the fly.io volume, indexed into a separate Chroma collection for semantic search.

**Rationale**:
- Markdown files are human-readable, portable, and easy to back up.
- A separate Chroma collection keeps explicit user-written thoughts distinct from auto-extracted conversation memories.

---

## 15. Portfolio Tracking (Planned)

**Decision (planned)**: Store holdings as a JSON file on the fly.io volume; fetch prices via yfinance (already installed).

**Rationale**: No broker OAuth or API keys needed — yfinance uses public Yahoo Finance data (~15 min delayed).

---

## 16. LangGraph Architecture (Separate)

**Decision**: LangGraph implementation exists in `graph.py` as a separate alternative.

**Rationale**: LangGraph provides a more structured approach to agent orchestration with built-in state management and checkpointing. The current simple/multi-turn implementations use manual message accumulation. LangGraph could replace these for more complex use cases.

**Status**: `graph.py` exists but `simple_agent.py` / `multi_turn_agent.py` are currently used in production.

---

## 17. Test Structure

**Decision**: Separate tests for simple and multi-turn agents.

**File structure**:
```
test_*.py
├── test_simple_agent.py      # Tests single-round behavior
├── test_multi_turn_agent.py # Tests iterative + comparison
└── test_agent.py             # Runner for both
```

**Key test case for multi-turn**: "Find the latest news about the stock market, then tell me how it affects Apple shares."
- This query REQUIRES multi-turn because:
  1. First need stock market news
  2. Then use that info to determine how it affects Apple
