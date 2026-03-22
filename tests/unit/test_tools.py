from unittest.mock import MagicMock, patch

from agent.tools.stock_lookup import GetStockTool, StockInput, _get_stock_impl
from agent.tools.weather_lookup import GetWeatherTool, WeatherInput, _get_weather_impl
from agent.tools.web_search import SearchInput, SearchWebTool, _search_web_impl


class TestWebSearchTool:
    def test_search_tool_initialization(self):
        tool = SearchWebTool()
        assert tool.name == "search_web"
        assert tool.args_schema == SearchInput

    def test_search_tool_run_with_mock(self):
        tool = SearchWebTool()
        with patch(
            "agent.tools.web_search._search_web_impl",
            return_value={
                "results": [
                    {
                        "title": "Test",
                        "url": "https://test.com",
                        "content": "Test content",
                    }
                ]
            },
        ):
            result = tool._run("test query")
            assert "Test" in result
            assert "test.com" in result

    def test_search_web_impl_error_handling(self):
        with (
            patch("agent.tools.web_search._tavily", None),
            patch("agent.tools.web_search.DDGS") as mock_ddgs,
        ):
            mock_ddgs.return_value.__enter__.return_value.news.side_effect = Exception(
                "Network error"
            )
            result = _search_web_impl("test")
            assert "error" in result


class TestStockTool:
    def test_stock_tool_initialization(self):
        tool = GetStockTool()
        assert tool.name == "get_stock"
        assert tool.args_schema == StockInput

    def test_stock_impl_success(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Apple Inc",
            "currentPrice": 150.0,
            "regularMarketChangePercent": 2.5,
            "marketCap": 1000000000,
            "fiftyTwoWeekHigh": 160.0,
            "fiftyTwoWeekLow": 140.0,
        }
        with patch("agent.tools.stock_lookup.yf.Ticker", return_value=mock_ticker):
            result = _get_stock_impl("AAPL")
            assert result["short_name"] == "Apple Inc"
            assert result["ticker"] == "AAPL"
            assert result["price"] == 150.0

    def test_stock_impl_error(self):
        with patch(
            "agent.tools.stock_lookup.yf.Ticker",
            side_effect=Exception("Invalid ticker"),
        ):
            result = _get_stock_impl("INVALID")
            assert "error" in result


class TestWeatherTool:
    def test_weather_tool_initialization(self):
        tool = GetWeatherTool()
        assert tool.name == "get_weather"
        assert tool.args_schema == WeatherInput

    def test_weather_impl_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current_condition": [
                {
                    "weatherDesc": [{"value": "Sunny"}],
                    "temp_C": "20",
                    "temp_F": "68",
                    "FeelsLikeC": "19",
                    "FeelsLikeF": "66",
                    "humidity": "60",
                    "windspeedKmph": "10",
                }
            ],
            "nearest_area": [
                {
                    "areaName": [{"value": "London"}],
                    "country": [{"value": "UK"}],
                }
            ],
        }
        with patch("agent.tools.weather_lookup.requests.get", return_value=mock_response):
            result = _get_weather_impl("London")
            assert result["city"] == "London"
            assert result["country"] == "UK"
            assert result["condition"] == "Sunny"

    def test_weather_impl_error(self):
        with patch(
            "agent.tools.weather_lookup.requests.get",
            side_effect=Exception("Network error"),
        ):
            result = _get_weather_impl("InvalidCity")
            assert "error" in result
