from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import logging
import os
import requests
import yfinance as yf
from duckduckgo_search import DDGS

try:
    from tavily import TavilyClient

    _tavily = (
        TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        if os.getenv("TAVILY_API_KEY")
        else None
    )
except ImportError:
    _tavily = None


logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    sources: Optional[list[str]] = Field(
        default=None, description="Optional list of domains to restrict search to"
    )
    topic: Optional[str] = Field(
        default=None, description="Optional topic filter: 'news' or 'general'"
    )


class StockInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, TSLA, MSFT")


class WeatherInput(BaseModel):
    location: str = Field(
        description="City name or location, e.g. 'London', 'New York'"
    )


def _search_web_impl(
    query: str, sources: list[str] | None = None, topic: str | None = None
) -> dict:
    """Searches the web using Tavily (primary) or DuckDuckGo (fallback)."""
    logger.info(f"Searching for: {query}")
    logger.info(f"[SEARCH] Tavily available: {_tavily is not None}")

    results_list = []

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
            logger.info(
                f"[SEARCH] Tavily returned {len(response.get('results', []))} results"
            )
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
            results = [r for r in ddgs.news(query, max_results=3)]
            if not results:
                results = [r for r in ddgs.text(query, max_results=3)]
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
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, query: str, sources: list[str] | None = None, topic: str | None = None
    ) -> str:
        result = _search_web_impl(query, sources, topic)
        if "error" in result:
            return f"Search failed: {result['error']}"

        formatted = []
        for r in result["results"]:
            formatted.append(f"- [{r['title']}]({r['url']})\n  {r['content']}")
        return "\n".join(formatted)


def _get_stock_impl(ticker: str) -> dict:
    """Fetches current price and key info for a stock ticker."""
    logger.info(f"Fetching stock data for: {ticker}")
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return {
            "short_name": info.get("shortName", ticker),
            "ticker": ticker.upper(),
            "price": price,
            "change_percent": info.get("regularMarketChangePercent", 0),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        logger.error(f"Stock fetch failed: {e}")
        return {"error": f"Could not fetch data for '{ticker}': {e}"}


class GetStockTool(BaseTool):
    name: str = "get_stock"
    description: str = (
        "Get current stock price and key financial info for a ticker symbol"
    )
    args_schema: Type[BaseModel] = StockInput

    def _run(self, ticker: str) -> str:
        result = _get_stock_impl(ticker)
        if "error" in result:
            return result["error"]

        return (
            f"{result['short_name']} ({result['ticker']})\n"
            f"Price: ${result['price']}\n"
            f"Change: {result['change_percent']:.2f}%\n"
            f"Market Cap: ${result['market_cap']:,}\n"
            f"52w High: ${result['fifty_two_week_high']} | Low: ${result['fifty_two_week_low']}"
        )


def _get_weather_impl(location: str) -> dict:
    """Fetches current weather for a location."""
    logger.info(f"Fetching weather for: {location}")
    try:
        resp = requests.get(
            f"https://wttr.in/{requests.utils.quote(location)}",
            params={"format": "j1"},
            timeout=(5, 20),
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        return {
            "city": city,
            "country": country,
            "condition": current["weatherDesc"][0]["value"],
            "temp_c": current["temp_C"],
            "temp_f": current["temp_F"],
            "feels_like_c": current["FeelsLikeC"],
            "feels_like_f": current["FeelsLikeF"],
            "humidity": current["humidity"],
            "wind_speed": current["windspeedKmph"],
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return {"error": f"Could not fetch weather for '{location}': {e}"}


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather conditions for a city or location"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        result = _get_weather_impl(location)
        if "error" in result:
            return result["error"]

        return (
            f"Weather in {result['city']}, {result['country']}:\n"
            f"Condition: {result['condition']}\n"
            f"Temp: {result['temp_c']}°C / {result['temp_f']}°F\n"
            f"Feels like: {result['feels_like_c']}°C / {result['feels_like_f']}°F\n"
            f"Humidity: {result['humidity']}% | Wind: {result['wind_speed']} km/h"
        )


TOOLS = [
    SearchWebTool(),
    GetStockTool(),
    GetWeatherTool(),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}
