# PersonalAssist Migration Plan

## Goals
1. Migrate from custom agentic loop to **LangGraph** for orchestration
2. Implement **structured responses** (text + sources + metadata)
3. Format responses as **HTML** for Telegram mobile display

---

## Phase 1: LangGraph Architecture

### 1.1 Dependencies
Add to `pyproject.toml`:
```python
langgraph >= 0.2.0
langchain-core >= 0.3.0
langchain-openai >= 0.2.0
pydantic >= 2.0
```

### 1.2 New File: `agent/graph.py`
```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph State                         │
├─────────────────────────────────────────────────────────────┤
│  messages: List[HumanMessage | AIMessage | ToolMessage]     │
│  user_query: str                                            │
│  tool_calls: List[dict]        # Track for structured out  │
│  sources: List[str]            # Source URLs                │
│  final_response: str           # Generated response        │
│  metadata: dict                 # Latency, iteration count   │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Nodes
| Node | Function |
|------|----------|
| `classify_node` | Determine if tool calling needed |
| `tools_node` | Execute tools (LangGraph tool executor) |
| `respond_node` | Generate final response with LLM |
| `format_html_node` | Convert to HTML |

### 1.4 Edges
```
START → classify
classify → respond (no tools needed)
classify → tools (tools needed)
tools → respond (max iterations or no more tools)
respond → format_html → END
```

---

## Phase 2: Structured Response

### 2.1 Response Schema (Pydantic)
```python
class AgentResponse(BaseModel):
    text: str                    # Final response text
    sources: List[Source]         # Source URLs with titles
    metadata: ResponseMetadata   # Latency, iterations, etc.

class Source(BaseModel):
    title: str
    url: str
    snippet: str

class ResponseMetadata(BaseModel):
    tool_calls: int
    total_latency_ms: float
    iteration_count: int
    memory_used: bool
```

### 2.2 Implementation
- Use `langchain.output_parsers` or Pydantic output parser
- Stream response to Telegram as HTML

---

## Phase 3: HTML Formatting

### 3.1 Telegram HTML Support
Limited to: `<b>`, `<i>`, `<u>`, `<code>`, `<pre>`, `<a>`

### 3.2 HTML Template
```html
<b>Response Title</b>

Response text with <i>emphasis</i> and <code>code</code>

<b>Sources:</b>
• <a href="url">Title</a>
• <a href="url">Title</a>

<i>Response time: 1.2s | Tools: 2</i>
```

### 3.3 Implementation (`response_formatter.py`)
- Convert Markdown → HTML (limited subset)
- Escape special chars: `<` → `&lt;`, `>` → `&gt;`
- Add styling classes for mobile readability

---

## Phase 4: Checkpointing (Memory)

### 4.1 LangGraph Checkpointing
- Use `MemorySaver` for in-memory checkpointing
- Or `SqliteSaver` for persistent state
- Remove mem0 dependency (or keep for long-term, use checkpointing for session)

### 4.2 State Management
- Persist conversation state per chat_id
- Load state on each message
- Clear on conversation end (optional)

---

## Phase 5: Code Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `agent/__init__.py` | Agent exports |
| `agent/graph.py` | LangGraph state graph |
| `agent/nodes.py` | Individual nodes |
| `agent/tools.py` | LangGraph tool definitions |
| `schemas/response.py` | Pydantic response schemas |

### Modified Files
| File | Changes |
|------|---------|
| `main.py` | Replace orchestrator with LangGraph agent invocation |
| `tools.py` | Refactor to LangGraph `@tool` decorators |
| `memory.py` | Replace with checkpointing (or remove) |
| `response_formatter.py` | Add HTML generation |
| `pyproject.toml` | Add LangGraph dependencies |

### Removed
- `orchestrator.py` (replaced by LangGraph)
- Potentially `memory.py` if using checkpointing only

---

## Implementation Order

1. **Week 1**: Setup LangGraph, define state schema, basic graph
2. **Week 2**: Tool definitions, node implementations
3. **Week 3**: Structured response with Pydantic
4. **Week 4**: HTML formatter, integrate with main.py
5. **Week 5**: Checkpointing, testing, cleanup

---

## Risks & Considerations

- **Latency**: LangGraph adds overhead (~50-100ms per node)
- **Complexity**: More code initially, but better maintainability
- **Telegram HTML**: Limited formatting, test thoroughly on mobile
- **Backward Compatibility**: Keep old code until new is proven
