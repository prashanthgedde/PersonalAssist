import logging
import os

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient

    _tavily = (
        TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
    )
except ImportError:
    _tavily = None

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    sources: list[str] | None = Field(
        default=None, description="Optional list of domains to restrict search to"
    )
    topic: str | None = Field(
        default=None, description="Optional topic filter: 'news' or 'general'"
    )


def _search_web_impl(
    query: str, sources: list[str] | None = None, topic: str | None = None
) -> dict:
    """Searches the web using Tavily (primary) or DuckDuckGo (fallback)."""
    logger.info(f"Searching for: {query}")
    logger.info(f"[SEARCH] Tavily available: {_tavily is not None}")

    if _tavily:
        try:
            kwargs = {"max_results": 5}
            if sources:
                kwargs["include_domains"] = sources
            if topic:
                kwargs["topic"] = topic
            logger.info(
                f"[SEARCH] Calling Tavily API: query='{query}', sources={sources}, topic={topic}"
            )
            response = _tavily.search(query, **kwargs)
            logger.info(f"[SEARCH] Tavily returned {len(response.get('results', []))} results")
            results = response.get("results", [])
            return {
                "results": [
                    {
                        "title": r["title"],
                        "url": r["url"],
                        "content": r["content"][:200],
                    }
                    for r in results
                ]
            }
        except Exception as e:
            logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=3))
            if not results:
                results = list(ddgs.text(query, max_results=3))
            return {
                "results": [
                    {
                        "title": r["title"],
                        "url": r.get("href", ""),
                        "content": r.get("body", r.get("snippet", ""))[:200],
                    }
                    for r in results
                ]
            }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e)}


class SearchWebTool(BaseTool):
    name: str = "search_web"
    description: str = (
        "Search the web for news, current events, Reddit discussions, YouTube videos, or any topic. "
        "Optionally restrict to specific sources like reddit.com, x.com, youtube.com, techcrunch.com."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, sources: list[str] | None = None, topic: str | None = None) -> str:
        result = _search_web_impl(query, sources, topic)
        if "error" in result:
            return f"Search failed: {result['error']}"

        formatted = []
        for r in result["results"]:
            formatted.append(f"- [{r['title']}]({r['url']})\n  {r['content']}")
        return "\n".join(formatted)
