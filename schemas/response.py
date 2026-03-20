from typing import Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""


class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: str
    latency_ms: float
    sources: list[str] = Field(default_factory=list)


class ResponseMetadata(BaseModel):
    tool_calls: int = 0
    total_latency_ms: float = 0.0
    iteration_count: int = 0
    memory_used: bool = False
    classification: str = "simple"


class AgentResponse(BaseModel):
    text: str
    sources: list[Source] = Field(default_factory=list)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    html: Optional[str] = None

    def to_html(self) -> str:
        if self.html:
            return self.html

        parts = []

        parts.append(self.text)

        if self.sources:
            parts.append("")
            parts.append("<b>Sources:</b>")
            for source in self.sources:
                parts.append(f'• <a href="{source.url}">{source.title}</a>')

        meta_parts = []
        if self.metadata.tool_calls > 0:
            meta_parts.append(f"Tools: {self.metadata.tool_calls}")
        if self.metadata.total_latency_ms > 0:
            meta_parts.append(f"Time: {self.metadata.total_latency_ms / 1000:.1f}s")

        if meta_parts:
            parts.append("")
            parts.append(f"<i>{' | '.join(meta_parts)}</i>")

        return "\n".join(parts)
