import json
import logging
from dataclasses import dataclass, field
from time import time


@dataclass
class ToolCall:
    """Tracks a single tool invocation."""
    name: str
    args: dict
    result_len: int
    sources: list = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ResponseSummary:
    """Tracks all metrics for a single user message response."""
    chat_id: int
    query: str
    start_time: float = field(default_factory=time)
    classification: str = ""
    agentic_loops: int = 0
    tools_called: list[ToolCall] = field(default_factory=list)
    lm_response_len: int = 0
    formatting: str = ""
    final_response_len: int = 0
    status: str = "pending"
    error: str = ""

    def elapsed_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        return round((time() - self.start_time) * 1000, 1)

    def add_tool_call(self, name: str, args: dict, result: str, sources: list = None, latency_ms: float = 0.0):
        """Record a tool call with its result."""
        self.tools_called.append(ToolCall(
            name=name,
            args=args,
            result_len=len(result) if result else 0,
            sources=sources or [],
            latency_ms=latency_ms
        ))

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query[:100],  # Truncate for readability
            "classification": self.classification,
            "agentic_loops": self.agentic_loops,
            "tools": [
                {
                    "name": tc.name,
                    "sources": tc.sources if tc.sources else None,
                    "result_len": tc.result_len,
                    "latency_ms": tc.latency_ms
                }
                for tc in self.tools_called
            ],
            "lm_response_len": self.lm_response_len,
            "formatting": self.formatting,
            "final_response_len": self.final_response_len,
            "total_latency_ms": self.elapsed_ms(),
            "status": self.status,
            "error": self.error if self.error else None
        }

    def log(self):
        """Log the summary as structured JSON."""
        logging.info(f"[SUMMARY] {json.dumps(self.to_dict())}")
