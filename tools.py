import logging
import os
import requests
import yfinance as yf
from duckduckgo_search import DDGS

try:
    from tavily import TavilyClient
    _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
except ImportError:
    _tavily = None


def search_web(query: str, sources: list[str] | None = None) -> str:
    """Searches the web using Tavily (primary) or DuckDuckGo (fallback)."""
    logging.info(f"Searching for: {query}")

    if _tavily:
        try:
            kwargs = {"max_results": 5}
            if sources:
                kwargs["include_domains"] = sources
            response = _tavily.search(query, **kwargs)
            results = response.get("results", [])
            return "\n".join([
                f"- [{r['title']}]({r['url']})\n  {r['content'][:200]}"
                for r in results
            ])
        except Exception as e:
            logging.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")

    # DuckDuckGo fallback
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.news(query, max_results=3)]
            if not results:
                results = [r for r in ddgs.text(query, max_results=3)]
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            return "\n".join([f"- {r['title']} ({r['date']}): {r['body']}" for r in results])
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return f"Search failed: {e}"


def get_stock(ticker: str) -> str:
    """Fetches current price and key info for a stock ticker."""
    logging.info(f"Fetching stock data for: {ticker}")
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return (
            f"{info.get('shortName', ticker)} ({ticker.upper()})\n"
            f"Price: ${price}\n"
            f"Change: {info.get('regularMarketChangePercent', 0):.2f}%\n"
            f"Market Cap: ${info.get('marketCap', 'N/A'):,}\n"
            f"52w High: ${info.get('fiftyTwoWeekHigh', 'N/A')} | Low: ${info.get('fiftyTwoWeekLow', 'N/A')}"
        )
    except Exception as e:
        logging.error(f"Stock fetch failed: {e}")
        return f"Could not fetch data for '{ticker}': {e}"


def get_weather(location: str) -> str:
    """Fetches current weather for a location."""
    logging.info(f"Fetching weather for: {location}")
    try:
        resp = requests.get(
            f"https://wttr.in/{requests.utils.quote(location)}",
            params={"format": "j1"},
            timeout=(5, 20)  # (connect, read)
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        return (
            f"Weather in {city}, {country}:\n"
            f"Condition: {current['weatherDesc'][0]['value']}\n"
            f"Temp: {current['temp_C']}°C / {current['temp_F']}°F\n"
            f"Feels like: {current['FeelsLikeC']}°C / {current['FeelsLikeF']}°F\n"
            f"Humidity: {current['humidity']}% | Wind: {current['windspeedKmph']} km/h"
        )
    except Exception as e:
        logging.error(f"Weather fetch failed: {e}")
        return f"Could not fetch weather for '{location}': {e}"


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for news, current events, Reddit discussions, YouTube videos, or any topic. "
                "Optionally restrict to specific sources like reddit.com, x.com, youtube.com, techcrunch.com."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domains to restrict search to, e.g. ['reddit.com', 'x.com']"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Get current stock price and key financial info for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, TSLA, MSFT"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city or location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location, e.g. 'London', 'New York'"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder to be sent to the user at a specific date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The reminder message"},
                    "remind_at": {"type": "string", "description": "ISO 8601 datetime string e.g. 2026-03-05T15:00:00"}
                },
                "required": ["message", "remind_at"]
            }
        }
    }
]
