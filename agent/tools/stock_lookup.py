import logging

import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StockInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, TSLA, MSFT")


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
    args_schema: type[BaseModel] = StockInput

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
