from agent.tools.stock_lookup import GetStockTool
from agent.tools.weather_lookup import GetWeatherTool
from agent.tools.web_search import SearchWebTool

TOOLS = [
    SearchWebTool(),
    GetStockTool(),
    GetWeatherTool(),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}

__all__ = ["TOOLS", "TOOL_MAP", "SearchWebTool", "GetStockTool", "GetWeatherTool"]
